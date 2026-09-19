"""3D visualization helpers for the Streamlit app.

Two renderers, chosen for what each is good at (both are real, tested, and
run fully offline — no browser plugins or GPU display server required):

- Plotly `Scatter3d` for the raw-mocap tab: skeletons/marker clouds are
  just points and line segments, and Plotly gives free, smooth,
  client-side orbit/zoom without any server round-trip per frame.
- `pyrender` offscreen rendering for the SMPL-X fit tab: a real lit,
  shaded 3D mesh render (the user asked for pyrender specifically), which
  Plotly's `Mesh3d` cannot match visually for a 10k-vertex body mesh.
  Verified to work headlessly on this machine; if a machine has no usable
  OpenGL backend at all, `render_mesh_frame` raises a clear `RuntimeError`
  rather than crashing the whole app.

Threading note (important on Windows): an OpenGL context is thread-affine,
and on Windows specifically, pyglet/PyOpenGL cannot even *create a fresh*
context on a thread other than the one that bootstrapped the process's
first context — a background thread's attempt fails with
"wglChoosePixelFormatARB is not exported", not just a context-reuse error.
Streamlit runs each script rerun on its own worker thread (not the
process's main thread), so a naive per-call or per-thread renderer breaks
under real `streamlit run` even though it works fine under direct-call
tests. The fix here is `_RenderWorker`: a single dedicated background
thread, started once and reused for the process's entire lifetime, which
owns the one true pyrender context; every `render_mesh_frame` call — from
whichever Streamlit thread — is marshaled onto that one thread via a queue
and blocks for its result. This was verified against the actual failure
mode (see the project chat log) before landing, not assumed to work.
"""
from __future__ import annotations

import queue
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Skeleton bone connectivity, from the real SMPL-X kinematic tree.
# ---------------------------------------------------------------------------
def skeleton_edges_from_parents(parents: Sequence[int]) -> List[Tuple[int, int]]:
    """(child, parent) pairs for every joint with a real parent (skips the
    root, whose parent index is -1)."""
    return [(child, int(parent)) for child, parent in enumerate(parents) if parent >= 0]


# ---------------------------------------------------------------------------
# Plotly: raw skeleton / marker-cloud playback
# ---------------------------------------------------------------------------
def compute_axis_bounds(points_over_time: np.ndarray, pad_ratio: float = 0.15):
    """A single fixed axis range covering every frame, so scrubbing the
    slider doesn't rescale/jitter the camera each frame."""
    flat = points_over_time.reshape(-1, 3)
    finite = flat[np.isfinite(flat).all(axis=1)]
    if finite.size == 0:
        return (-1, 1), (-1, 1), (-1, 1)
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * pad_ratio
    lo, hi = lo - pad, hi + pad
    # Equal aspect: expand every axis to the largest span, centered.
    max_span = float((hi - lo).max())
    center = (lo + hi) / 2
    half = max_span / 2
    lo, hi = center - half, center + half
    return (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])


def plot_points_figure(
    points: np.ndarray,
    conf: Optional[np.ndarray] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    labels: Optional[List[str]] = None,
    bounds=None,
    title: str = "",
    marker_color: str = "#2F5496",
    unobserved_color: str = "#C00000",
) -> go.Figure:
    """One frame of a point cloud (marker set or skeleton joints), with
    optional bone edges and per-point observed/unobserved coloring — makes
    the "absent, not measured" distinction visible, not just numeric."""
    points = np.asarray(points)
    k = points.shape[0]
    if conf is None:
        conf = np.ones(k)
    colors = [marker_color if c > 0 else unobserved_color for c in conf]
    text = labels if labels is not None else [str(i) for i in range(k)]

    fig = go.Figure()

    if edges:
        xs, ys, zs = [], [], []
        for a, b in edges:
            if a < k and b < k:
                xs += [points[a, 0], points[b, 0], None]
                ys += [points[a, 1], points[b, 1], None]
                zs += [points[a, 2], points[b, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#8C8C8C", width=4), hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=5, color=colors),
        text=text, hoverinfo="text", showlegend=False,
    ))

    scene = dict(
        xaxis=dict(title="x", visible=True),
        yaxis=dict(title="y", visible=True),
        zaxis=dict(title="z", visible=True),
        aspectmode="cube",
    )
    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        scene["xaxis"]["range"] = [xlo, xhi]
        scene["yaxis"]["range"] = [ylo, yhi]
        scene["zaxis"]["range"] = [zlo, zhi]

    fig.update_layout(
        scene=scene, title=title, margin=dict(l=0, r=0, t=30, b=0),
        height=520,
    )
    return fig


def _points_traces_dict(points: np.ndarray, colors, text,
                         edge_a: np.ndarray, edge_b: np.ndarray) -> List[dict]:
    """The two traces (edges, markers) for one frame, as plain dicts —
    always in this fixed order and shape, since Plotly animation frames
    update traces positionally (every frame of an animated figure must
    emit the same trace count/order, including an empty edges trace).

    Plain dicts, not `go.Scatter3d` objects: constructing and validating a
    `go.Scatter3d`/`go.Frame` object per frame is the dominant cost for a
    figure with hundreds-to-thousands of frames (measured: ~11s to build
    one real 2751-frame capture's animation using `go.*` objects
    throughout). Plotly ultimately serializes everything to these same
    JSON-shaped dicts anyway; building them directly and validating the
    whole figure exactly once at the end (via a single `go.Figure(...)`
    call) skips that many-times-repeated validation overhead entirely.
    """
    if edge_a.size:
        ex = np.stack([points[edge_a, 0], points[edge_b, 0], np.full(edge_a.shape, np.nan)], axis=1).ravel()
        ey = np.stack([points[edge_a, 1], points[edge_b, 1], np.full(edge_a.shape, np.nan)], axis=1).ravel()
        ez = np.stack([points[edge_a, 2], points[edge_b, 2], np.full(edge_a.shape, np.nan)], axis=1).ravel()
    else:
        ex = ey = ez = np.array([])

    edge_trace = dict(type="scatter3d", x=ex, y=ey, z=ez, mode="lines",
                       line=dict(color="#8C8C8C", width=4), hoverinfo="skip", showlegend=False)
    marker_trace = dict(type="scatter3d", x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                         marker=dict(size=5, color=colors), text=text, hoverinfo="text", showlegend=False)
    return [edge_trace, marker_trace]


def plot_points_animation(
    points_seq: np.ndarray,
    conf_seq: Optional[np.ndarray] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    labels: Optional[List[str]] = None,
    bounds=None,
    fps: float = 30.0,
    title: str = "",
    marker_color: str = "#2F5496",
    unobserved_color: str = "#C00000",
) -> go.Figure:
    """A single Plotly figure with every frame of `points_seq` (T, K, 3)
    baked in as a native Plotly animation (`frames` + Play/Pause buttons +
    a scrub slider), so playback runs entirely in the browser's own
    JavaScript — Streamlit's Python script executes exactly once to build
    and send this figure, and never runs again for the rest of playback.

    This replaces an earlier Python-driven approach (a `st.session_state`
    + `st.rerun()` loop advancing one frame per script execution) that
    still measurably interfered with scrolling and with clicking a
    chart's "view fullscreen" control while "Play" was active — even
    though each individual rerun was fast and bounded, the backend was
    continuously busy back-to-back for the whole playthrough, and
    Streamlit's own status/interaction handling during that window is
    what was fighting the user's scroll and click input (reported bug,
    persisting after the rerun-based fix). A native Plotly animation has
    no such window: after this one figure is sent, the Python side is
    completely idle, so the page is exactly as scrollable and interactive
    as it is when nothing is playing at all.
    """
    t = points_seq.shape[0]
    k = points_seq.shape[1]
    if conf_seq is None:
        conf_seq = np.ones(points_seq.shape[:2])

    text = labels if labels is not None else [str(i) for i in range(k)]

    # Valid edge endpoints, precomputed once — never changes across frames
    # (only the endpoint *positions* do).
    if edges:
        valid = [(a, b) for a, b in edges if a < k and b < k]
        edge_a = np.array([a for a, b in valid], dtype=np.int64)
        edge_b = np.array([b for a, b in valid], dtype=np.int64)
    else:
        edge_a = edge_b = np.array([], dtype=np.int64)

    # Per-point observed/unobserved coloring is, in practice, identical
    # across every frame for every format except video (MediaPipe is the
    # only source with real per-frame confidence) — recomputing an
    # identical Python list of color strings on each of possibly
    # thousands of frames is pure waste, so it's done once when possible.
    conf_is_constant = bool(np.all(conf_seq == conf_seq[0:1]))
    const_colors = (
        [marker_color if c > 0 else unobserved_color for c in conf_seq[0]] if conf_is_constant else None
    )

    frame_duration_ms = int(1000.0 / max(fps, 1.0))
    frames = []
    for i in range(t):
        colors = const_colors if conf_is_constant else \
            [marker_color if c > 0 else unobserved_color for c in conf_seq[i]]
        frames.append(dict(data=_points_traces_dict(points_seq[i], colors, text, edge_a, edge_b), name=str(i)))

    scene = dict(
        xaxis=dict(title="x", visible=True),
        yaxis=dict(title="y", visible=True),
        zaxis=dict(title="z", visible=True),
        aspectmode="cube",
    )
    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        scene["xaxis"]["range"] = [xlo, xhi]
        scene["yaxis"]["range"] = [ylo, yhi]
        scene["zaxis"]["range"] = [zlo, zhi]

    layout = dict(
        scene=scene, title=title, margin=dict(l=0, r=0, t=30, b=0), height=560,
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.08, xanchor="left", yanchor="top",
            showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate", args=[
                    None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                               fromcurrent=True, mode="immediate",
                               transition=dict(duration=0)),
                ]),
                dict(label="⏸ Pause", method="animate", args=[
                    [None], dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                ]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.12, y=-0.08, len=0.88, xanchor="left", yanchor="top",
            currentvalue=dict(prefix="Frame: ", visible=True),
            steps=[
                dict(method="animate", label=str(i),
                     args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                for i in range(t)
            ],
        )],
    )

    # One single validation pass over the whole figure (data + frames +
    # layout together), instead of one per go.Scatter3d/go.Frame object —
    # this is what actually fixes the ~11s build time measured on a real
    # 2751-frame capture (see this function's docstring).
    return go.Figure(data=frames[0]["data"] if frames else [], layout=layout, frames=frames)


# ---------------------------------------------------------------------------
# pyrender: SMPL-X mesh offscreen rendering
# ---------------------------------------------------------------------------
class _RenderWorker:
    """One background thread, started once, that owns every pyrender
    OffscreenRenderer for the process's lifetime.

    Confirmed necessary on this machine, not a precaution: creating a
    renderer on a second thread — even a brand-new one that never touches
    another thread's renderer — fails outright with
    "wglChoosePixelFormatARB is not exported", because on Windows the WGL
    extension-loading pyglet relies on can only bootstrap on whichever
    thread first created a GL context in this process. Routing every call
    through one persistent thread sidesteps that entirely.
    """

    def __init__(self):
        self._jobs: "queue.Queue" = queue.Queue()
        # Deliberately only ONE live (size, renderer) pair, not a cache
        # keyed by every size ever requested: pyglet on Windows tries to
        # share GL objects between windows/contexts it creates within the
        # same process, and that negotiation can fail outright
        # ("Unable to share contexts") the moment a second differently-sized
        # OffscreenRenderer is created while an earlier one is still alive
        # — confirmed by this module's own test suite hitting exactly that
        # the first time two render sizes were exercised back to back. The
        # image-size selector in the UI changes size occasionally, not per
        # frame, so paying a fresh ~1.4s context-creation cost on a size
        # change (rather than caching every size forever) is the right
        # trade-off.
        self._renderer = None
        self._renderer_size = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="pyrender-worker")
        self._thread.start()

    def _run(self):
        while True:
            func, args, kwargs, result_q = self._jobs.get()
            try:
                result_q.put(("ok", func(self, *args, **kwargs)))
            except Exception as e:  # noqa: BLE001 - forwarded to the caller's thread
                result_q.put(("error", e))

    def _get_renderer(self, width: int, height: int):
        import pyrender

        key = (width, height)
        if self._renderer_size != key:
            if self._renderer is not None:
                self._renderer.delete()
            self._renderer = pyrender.OffscreenRenderer(width, height)
            self._renderer_size = key
        return self._renderer

    def submit(self, func, *args, timeout: float = 30.0, **kwargs):
        result_q: "queue.Queue" = queue.Queue(maxsize=1)
        self._jobs.put((func, args, kwargs, result_q))
        status, payload = result_q.get(timeout=timeout)
        if status == "error":
            raise payload
        return payload


_worker: Optional[_RenderWorker] = None
_worker_lock = threading.Lock()


def _get_worker() -> _RenderWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:  # re-check inside the lock (double-checked init)
                _worker = _RenderWorker()
    return _worker


def render_mesh_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth_deg: float = 20.0,
    elevation_deg: float = 10.0,
    distance: Optional[float] = None,
    image_size: Tuple[int, int] = (480, 480),
    mesh_color=(0.65, 0.74, 0.86, 1.0),
    up_axis: str = "z",
) -> np.ndarray:
    """Render one SMPL-X mesh with pyrender, camera orbiting the body
    center. Returns an (H, W, 3) uint8 RGB image.

    Safe to call from any thread — the actual OpenGL work always runs on
    the single dedicated render-worker thread (see `_RenderWorker`); this
    function just submits the job and blocks for the result.

    `up_axis`: which world axis is "up" for the *vertices being passed in*.
    Every vertex array this app renders comes from `body.forward()` fed
    with trans/global_orient taken either straight from AMASS-convention
    mocap parameters (Tier 0 .npz) or from the solver's fitted pose (whose
    global_orient was itself solved against Z-up mocap/marker observations
    — see doc/Mocap_Unification_Design.docx section 6.4). That makes Z, not
    Y, the vertical axis for this pipeline's data, even though SMPL-X's own
    *unposed* rest frame is locally Y-up (see tests/test_body.py) — the two
    are different frames and must not be confused. Getting this wrong
    doesn't move a single vertex, but it does aim the camera's "up" at the
    wrong axis, which is exactly why the previous hardcoded Y-up camera
    made every render look tipped onto its side.

    Raises RuntimeError (not a generic exception) if this machine has no
    usable offscreen OpenGL backend, so the caller can show a clear
    message instead of a stack trace.
    """
    if up_axis not in ("y", "z"):
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")

    worker = _get_worker()
    return worker.submit(
        _render_mesh_frame_impl, vertices, faces, azimuth_deg, elevation_deg,
        distance, image_size, mesh_color, up_axis,
    )


def _render_mesh_frame_impl(
    worker: "_RenderWorker",
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance: Optional[float],
    image_size: Tuple[int, int],
    mesh_color,
    up_axis: str,
) -> np.ndarray:
    """The actual render, always executed on `_RenderWorker`'s thread."""
    import trimesh
    import pyrender

    up_vec = np.array([0.0, 1.0, 0.0]) if up_axis == "y" else np.array([0.0, 0.0, 1.0])

    try:
        center = vertices.mean(axis=0)
        extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        if distance is None:
            distance = max(extent * 1.6, 1.5)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = np.tile(
            (np.array(mesh_color) * 255).astype(np.uint8), (vertices.shape[0], 1)
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        scene = pyrender.Scene(bg_color=[0.94, 0.94, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.38])
        scene.add(py_mesh)

        # Spherical offset around `center`: azimuth sweeps the horizontal
        # plane perpendicular to up_vec, elevation tilts toward up_vec.
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        if up_axis == "z":
            offset = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            fill_offset = np.array([-0.6, 0.6, 0.8])
        else:
            offset = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
            fill_offset = np.array([-0.6, 0.8, 0.6])
        cam_pos = center + distance * offset
        cam_pose = _look_at(cam_pos, center, up=up_vec)
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(40.0), aspectRatio=image_size[0] / image_size[1])
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=cam_pose)
        fill_pose = _look_at(center + distance * fill_offset, center, up=up_vec)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill_pose)

        renderer = worker._get_renderer(*image_size)
        color, _depth = renderer.render(scene)
        return color
    except Exception as e:  # pragma: no cover - depends on the host's GL stack
        raise RuntimeError(
            "pyrender could not produce an offscreen render on this machine "
            f"({type(e).__name__}: {e}). This usually means no OpenGL-capable "
            "backend (EGL/OSMesa/pyglet window) is available in this "
            "environment."
        ) from e


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world pose matrix for pyrender (camera looks down -Z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


# ---------------------------------------------------------------------------
# Client-side player for the pre-rendered pyrender mesh frames.
# ---------------------------------------------------------------------------
def build_mesh_player_html(images_png: List[bytes], fps: float, width: int, height: int) -> str:
    """A self-contained HTML/JS <img> player for a list of pre-rendered
    mesh frames — Play/Pause, a scrub slider, and its own Fullscreen
    button, animated entirely by the browser via `setInterval` once this
    HTML is sent to it.

    Why this exists instead of driving playback through repeated
    `st.image()` calls: any approach that needs Python to re-run in order
    to show the next frame (a `st.session_state` + `st.rerun()` loop, one
    frame per script execution) keeps Streamlit's backend continuously
    busy for the whole playthrough. That measurably interfered with the
    page's scrolling and with clicking a chart/image's native "view
    fullscreen" control (reported bug, persisting even after making that
    loop fast and bounded) — the backend being busy back-to-back the
    entire time is itself the problem, regardless of how quickly each
    individual step completes. Pre-rendering every frame once (the only
    Python work involved) and handing the finished images to a plain
    client-side animation loop means Python is completely idle for the
    rest of playback, so the page behaves exactly as it does when nothing
    is playing — scrolling and every other interaction included.

    A dedicated Fullscreen button is included (via the browser's own
    Fullscreen API) rather than relying on Streamlit's native per-element
    fullscreen icon, since that icon only decorates Streamlit's own
    built-in elements (`st.image`, `st.plotly_chart`, ...), not an
    embedded `components.v1.html` block like this one.
    """
    import base64

    b64_frames = [base64.b64encode(png).decode("ascii") for png in images_png]
    frame_count = len(b64_frames)
    interval_ms = int(1000.0 / max(fps, 1.0))
    frames_json = "[" + ",".join(f'"data:image/png;base64,{b}"' for b in b64_frames) + "]"

    return f"""
<div id="lmv-root" style="font-family:sans-serif;background:#f5f5f7;border-radius:8px;padding:10px;">
  <div id="lmv-stage" style="display:flex;justify-content:center;align-items:center;
       background:#eceef1;border-radius:6px;overflow:hidden;">
    <img id="lmv-img" src="" style="max-width:100%;height:auto;display:block;" />
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap;">
    <button id="lmv-play" style="padding:4px 12px;cursor:pointer;">&#9654; Play</button>
    <button id="lmv-pause" style="padding:4px 12px;cursor:pointer;" disabled>&#9208; Pause</button>
    <button id="lmv-full" style="padding:4px 12px;cursor:pointer;">&#x26F6; Fullscreen</button>
    <input id="lmv-slider" type="range" min="0" max="{max(frame_count - 1, 0)}" value="0"
           style="flex:1;min-width:120px;" />
    <span id="lmv-label" style="min-width:70px;text-align:right;font-size:0.85em;color:#444;">
      1 / {frame_count}
    </span>
  </div>
</div>
<script>
(function() {{
  const frames = {frames_json};
  const n = frames.length;
  const img = document.getElementById("lmv-img");
  const slider = document.getElementById("lmv-slider");
  const label = document.getElementById("lmv-label");
  const playBtn = document.getElementById("lmv-play");
  const pauseBtn = document.getElementById("lmv-pause");
  const fullBtn = document.getElementById("lmv-full");
  const stage = document.getElementById("lmv-stage");
  const root = document.getElementById("lmv-root");
  let idx = 0;
  let timer = null;

  function show(i) {{
    idx = ((i % n) + n) % n;
    if (n > 0) {{ img.src = frames[idx]; }}
    slider.value = idx;
    label.textContent = (idx + 1) + " / " + n;
  }}

  function play() {{
    if (timer !== null || n <= 1) return;
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    timer = setInterval(function() {{
      if (idx >= n - 1) {{ pause(); return; }}
      show(idx + 1);
    }}, {interval_ms});
  }}

  function pause() {{
    if (timer !== null) {{ clearInterval(timer); timer = null; }}
    playBtn.disabled = false;
    pauseBtn.disabled = true;
  }}

  slider.addEventListener("input", function() {{ pause(); show(parseInt(slider.value, 10)); }});
  playBtn.addEventListener("click", play);
  pauseBtn.addEventListener("click", pause);
  fullBtn.addEventListener("click", function() {{
    if (!document.fullscreenElement) {{
      (root.requestFullscreen || root.webkitRequestFullscreen || function(){{}}).call(root);
    }} else {{
      (document.exitFullscreen || document.webkitExitFullscreen || function(){{}}).call(document);
    }}
  }});

  show(0);
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Interactive client-side viewer (three.js): real mouse-orbit camera, a
# static ground plane, and the mesh geometry itself streamed to the browser
# instead of being pre-rendered to flat images server-side.
# ---------------------------------------------------------------------------
def build_mesh_player_threejs(
    vertices_seq: np.ndarray,
    faces: np.ndarray,
    fps: float,
    up_axis: str = "z",
    width: int = 700,
    height: int = 560,
    mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
) -> str:
    """A self-contained three.js viewport for an animated SMPL-X mesh:
    drag-to-orbit / scroll-to-zoom / right-drag-to-pan camera (three.js
    `OrbitControls`) plus a fixed ground grid, Play/Pause/scrub/Fullscreen
    controls, and a Reset View button.

    Why this replaces the pyrender-based player for the app's SMPL-X tab:
    `render_mesh_frame`/`build_mesh_player_html` pre-bake one fixed camera
    angle (an azimuth/elevation slider set once, server-side) into flat
    PNGs, so the only way to "look around" was re-rendering every frame on
    the Python side from a new angle — there is no way to make that feel
    like a game/engine viewport where the *user's mouse* orbits the camera
    in real time. Sending the actual mesh geometry to the browser once and
    letting three.js's own WebGL renderer + OrbitControls handle the
    camera gives real, continuous mouse-driven orbiting with no server
    round-trip, plus a ground plane that is a real static object in the
    scene (so orbiting the camera around the character never "rotates the
    environment" the way re-rendering from a new azimuth could look like).

    `render_mesh_frame`/`build_mesh_player_html` are kept as-is (still
    tested, still a valid offline/no-WebGL fallback) — this is an
    additional, not a replacement, capability.

    `up_axis`: which axis is "up" for `vertices_seq` (see
    `render_mesh_frame`'s docstring for why this pipeline's fitted data is
    Z-up). three.js's own convention is Y-up, so z-up vertices are rotated
    once here via the same proper (determinant +1) rotation used
    everywhere else in this project (`src.smplx.ops.convert_up_axis`) —
    never a single-axis sign flip, which would mirror left/right.
    """
    import base64

    from src.smplx.ops import convert_up_axis

    if up_axis not in ("y", "z"):
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")

    v = np.asarray(vertices_seq, dtype=np.float32)
    if up_axis == "z":
        v = convert_up_axis(v, "z", "y").astype(np.float32)
    t = int(v.shape[0])
    n_verts = int(v.shape[1]) if t > 0 else 0
    faces_u32 = np.asarray(faces, dtype=np.uint32)

    if t > 0:
        flat = v.reshape(-1, 3)
        lo, hi = flat.min(axis=0), flat.max(axis=0)
        center = ((lo + hi) / 2).tolist()
        ground_y = float(lo[1])
        extent = float(np.linalg.norm(hi - lo))
    else:
        center = [0.0, 0.0, 0.0]
        ground_y = 0.0
        extent = 1.0
    ground_size = max(extent * 3.0, 2.0)
    distance = max(extent * 1.8, 1.5)

    verts_b64 = base64.b64encode(v.tobytes()).decode("ascii")
    faces_b64 = base64.b64encode(faces_u32.tobytes()).decode("ascii")
    r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in mesh_color)
    color_hex = f"0x{r:02x}{g:02x}{b:02x}"
    interval_ms = int(1000.0 / max(fps, 1.0))
    cx, cy, cz = center

    return f"""
<div id="tjs-root" style="font-family:sans-serif;background:#f5f5f7;border-radius:8px;padding:10px;">
  <div id="tjs-stage" style="width:{width}px;height:{height}px;margin:0 auto;
       background:#e9ebee;border-radius:6px;overflow:hidden;line-height:0;"></div>
  <div style="display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap;">
    <button id="tjs-play" style="padding:4px 12px;cursor:pointer;">&#9654; Play</button>
    <button id="tjs-pause" style="padding:4px 12px;cursor:pointer;" disabled>&#9208; Pause</button>
    <button id="tjs-reset" style="padding:4px 12px;cursor:pointer;">&#8635; Reset View</button>
    <button id="tjs-full" style="padding:4px 12px;cursor:pointer;">&#x26F6; Fullscreen</button>
    <input id="tjs-slider" type="range" min="0" max="{max(t - 1, 0)}" value="0"
           style="flex:1;min-width:120px;" />
    <span id="tjs-label" style="min-width:70px;text-align:right;font-size:0.85em;color:#444;">
      1 / {t}
    </span>
  </div>
  <div style="font-size:0.78em;color:#777;margin-top:4px;">
    Drag to orbit &middot; scroll to zoom &middot; right-drag (or two-finger drag) to pan
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {{
  const nVerts = {n_verts};
  const nFrames = {t};
  const width = {width};
  const height = {height};
  const groundY = {ground_y};
  const groundSize = {ground_size};
  const distance = {distance};
  const center = new THREE.Vector3({cx}, {cy}, {cz});
  const interval = {interval_ms};

  function b64ToBuffer(b64) {{
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {{ bytes[i] = bin.charCodeAt(i); }}
    return bytes.buffer;
  }}
  const allVerts = new Float32Array(b64ToBuffer("{verts_b64}"));
  const faces = new Uint32Array(b64ToBuffer("{faces_b64}"));

  const stage = document.getElementById("tjs-stage");
  const slider = document.getElementById("tjs-slider");
  const label = document.getElementById("tjs-label");
  const playBtn = document.getElementById("tjs-play");
  const pauseBtn = document.getElementById("tjs-pause");
  const resetBtn = document.getElementById("tjs-reset");
  const fullBtn = document.getElementById("tjs-full");
  const root = document.getElementById("tjs-root");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe9ebee);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 1000);
  function defaultCameraPos() {{
    return new THREE.Vector3(
      center.x + distance * 0.7, center.y + distance * 0.55, center.z + distance * 0.9
    );
  }}
  camera.position.copy(defaultCameraPos());

  const renderer = new THREE.WebGLRenderer({{antialias: true}});
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  stage.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = distance * 0.1;
  controls.maxDistance = distance * 6.0;
  controls.update();

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(center.x + distance, center.y + distance * 1.2, center.z + distance * 0.6);
  keyLight.castShadow = true;
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
  fillLight.position.set(center.x - distance, center.y + distance * 0.4, center.z - distance * 0.8);
  scene.add(fillLight);

  // Static ground: a real object in the scene, not a re-rendered
  // background — orbiting the camera around the character never moves it.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(groundSize, groundSize),
    new THREE.MeshStandardMaterial({{color: 0xd7d9dd, roughness: 1.0}})
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(center.x, groundY, center.z);
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(groundSize, 24, 0x9099a6, 0xbfc4cb);
  grid.position.set(center.x, groundY + 0.002, center.z);
  scene.add(grid);

  const geometry = new THREE.BufferGeometry();
  const posArray = new Float32Array(Math.max(nVerts * 3, 1));
  if (nFrames > 0) {{ posArray.set(allVerts.subarray(0, nVerts * 3)); }}
  const posAttr = new THREE.BufferAttribute(posArray, 3);
  geometry.setAttribute("position", posAttr);
  geometry.setIndex(new THREE.BufferAttribute(faces, 1));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({{color: {color_hex}, roughness: 0.55, metalness: 0.05}});
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  scene.add(mesh);

  let idx = 0;
  function showFrame(i) {{
    if (nFrames <= 0) return;
    idx = ((i % nFrames) + nFrames) % nFrames;
    const offset = idx * nVerts * 3;
    posAttr.array.set(allVerts.subarray(offset, offset + nVerts * 3));
    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();
    slider.value = idx;
    label.textContent = (idx + 1) + " / " + nFrames;
  }}

  let timer = null;
  function play() {{
    if (timer !== null || nFrames <= 1) return;
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    timer = setInterval(function() {{
      if (idx >= nFrames - 1) {{ pause(); return; }}
      showFrame(idx + 1);
    }}, interval);
  }}
  function pause() {{
    if (timer !== null) {{ clearInterval(timer); timer = null; }}
    playBtn.disabled = false;
    pauseBtn.disabled = true;
  }}

  slider.addEventListener("input", function() {{ pause(); showFrame(parseInt(slider.value, 10)); }});
  playBtn.addEventListener("click", play);
  pauseBtn.addEventListener("click", pause);
  resetBtn.addEventListener("click", function() {{
    camera.position.copy(defaultCameraPos());
    controls.target.copy(center);
    controls.update();
  }});
  fullBtn.addEventListener("click", function() {{
    if (!document.fullscreenElement) {{
      (root.requestFullscreen || root.webkitRequestFullscreen || function(){{}}).call(root);
    }} else {{
      (document.exitFullscreen || document.webkitExitFullscreen || function(){{}}).call(document);
    }}
  }});

  // The render loop runs continuously at the display's own rate so mouse
  // orbiting stays smooth regardless of the mocap clip's fps — only
  // `showFrame` (driven by the interval above) is paced to `fps`.
  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();
  showFrame(0);
}})();
</script>
"""
