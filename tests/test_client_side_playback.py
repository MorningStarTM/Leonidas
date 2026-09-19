"""Tests for the client-side (browser-driven) playback mechanisms that
replaced the earlier Python/`st.rerun()`-driven animation.

Background: an initial fix made the per-frame Streamlit rerun loop fast
and bounded, but the reported bug (page unscrollable, chart "view
fullscreen" unresponsive while "Play" was active) persisted — because the
backend was still continuously busy back-to-back for the whole playthrough,
which is what was fighting the browser's own interaction handling,
regardless of how quickly each individual step completed. The fix removes
Python from the playback loop entirely:

- `plot_points_animation` bakes every frame into one Plotly figure using
  Plotly's native `frames` + `updatemenus` + `sliders` animation API, so
  the browser's own JS drives playback with no server involvement at all.
- `build_mesh_player_html` pre-renders every mesh frame once (the only
  Python work needed) and hands the images to a small self-contained
  HTML/JS `<img>` player with its own Play/Pause/scrub/Fullscreen
  controls, again needing no further Python execution to advance frames.

These are tested directly here (structural/content assertions on the
Figure and HTML string) since Streamlit's `AppTest` harness has no typed
accessor for `st.components.v1.html` content or for a Plotly figure's
`frames`/`layout` internals.
"""
import base64

import numpy as np
import pytest

from app.render3d import build_mesh_player_html, build_mesh_player_threejs, plot_points_animation


def _synthetic_points(t=6, k=5):
    rng = np.random.default_rng(0)
    points = rng.normal(0, 1, (t, k, 3))
    conf = np.ones((t, k))
    if t > 2 and k > 1:
        conf[2, 1] = 0.0  # one unobserved point-frame, to exercise that path
    return points, conf


# ---------------------------------------------------------------------------
# plot_points_animation (native Plotly animation, raw-mocap tab)
# ---------------------------------------------------------------------------
def test_animation_has_one_frame_per_timestep():
    points, conf = _synthetic_points(t=6, k=5)
    fig = plot_points_animation(points, conf_seq=conf, edges=[(0, 1), (1, 2)],
                                 labels=[f"p{i}" for i in range(5)], fps=15.0)
    assert len(fig.frames) == 6
    assert [f.name for f in fig.frames] == [str(i) for i in range(6)]


def test_animation_has_play_pause_buttons():
    points, conf = _synthetic_points()
    fig = plot_points_animation(points, conf_seq=conf, fps=30.0)
    updatemenus = fig.layout.updatemenus
    assert len(updatemenus) == 1
    labels = [b.label for b in updatemenus[0].buttons]
    assert any("Play" in l for l in labels)
    assert any("Pause" in l for l in labels)
    # The Play button must actually carry an `animate` method — that's
    # what makes Plotly's own JS runtime drive playback client-side,
    # rather than this being an inert button.
    play_button = next(b for b in updatemenus[0].buttons if "Play" in b.label)
    assert play_button.method == "animate"


def test_animation_has_scrub_slider_matching_frame_count():
    points, conf = _synthetic_points(t=8, k=3)
    fig = plot_points_animation(points, conf_seq=conf, fps=30.0)
    sliders = fig.layout.sliders
    assert len(sliders) == 1
    assert len(sliders[0].steps) == 8


def test_animation_frame_duration_matches_fps():
    points, conf = _synthetic_points(t=3, k=2)
    fig = plot_points_animation(points, conf_seq=conf, fps=25.0)
    play_button = next(b for b in fig.layout.updatemenus[0].buttons if "Play" in b.label)
    # args = [None, {frame: {duration: ms, ...}, ...}]
    frame_settings = play_button.args[1]["frame"]
    assert frame_settings["duration"] == int(1000.0 / 25.0)


def test_animation_marks_unobserved_points_distinctly():
    points, conf = _synthetic_points(t=6, k=5)
    fig = plot_points_animation(points, conf_seq=conf, marker_color="#111111", unobserved_color="#EE0000")
    # Frame index 2 has an unobserved point at column 1 (see _synthetic_points).
    marker_trace = fig.frames[2].data[1]  # [edges_trace, markers_trace]
    colors = list(marker_trace.marker.color)
    assert colors[1] == "#EE0000"
    assert colors[0] == "#111111"


def test_animation_trace_order_is_stable_across_frames():
    """Plotly frames update traces positionally — every frame must emit
    the same trace count/order (edges, then markers), including when
    `edges` is empty, or the animation would silently corrupt itself."""
    points, conf = _synthetic_points(t=4, k=3)
    fig_with_edges = plot_points_animation(points, conf_seq=conf, edges=[(0, 1)])
    fig_without_edges = plot_points_animation(points, conf_seq=conf, edges=None)
    for fig in (fig_with_edges, fig_without_edges):
        for frame in fig.frames:
            assert len(frame.data) == 2  # always [edges_trace, markers_trace]


def test_animation_handles_single_frame_without_error():
    points, conf = _synthetic_points(t=1, k=4)
    fig = plot_points_animation(points, conf_seq=conf)
    assert len(fig.frames) == 1


# ---------------------------------------------------------------------------
# build_mesh_player_html (client-side HTML/JS player, SMPL-X fit tab)
# ---------------------------------------------------------------------------
def _fake_pngs(n=4):
    # Minimal valid 1x1 PNG bytes, repeated — content doesn't matter for
    # these structural tests, only that each is distinctly identifiable.
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    return [tiny_png for _ in range(n)]


def test_mesh_player_embeds_every_frame():
    pngs = _fake_pngs(5)
    html = build_mesh_player_html(pngs, fps=24.0, width=300, height=300)
    b64 = base64.b64encode(pngs[0]).decode("ascii")
    assert html.count(b64) == 5  # all 5 frames are identical bytes here, each embedded once
    assert "const frames = [" in html
    # Count actual data-URI entries in the JS array, not just substring occurrences.
    assert html.count("data:image/png;base64,") == 5


def test_mesh_player_has_playback_controls():
    html = build_mesh_player_html(_fake_pngs(3), fps=30.0, width=200, height=200)
    assert 'id="lmv-play"' in html
    assert 'id="lmv-pause"' in html
    assert 'id="lmv-slider"' in html
    assert "setInterval" in html


def test_mesh_player_has_dedicated_fullscreen_control():
    """Streamlit's own per-element fullscreen icon only decorates built-in
    elements (st.image, st.plotly_chart, ...), not an embedded
    components.v1.html block — this player must supply its own, via the
    browser's Fullscreen API."""
    html = build_mesh_player_html(_fake_pngs(3), fps=30.0, width=200, height=200)
    assert 'id="lmv-full"' in html
    assert "requestFullscreen" in html


def test_mesh_player_interval_matches_fps():
    html = build_mesh_player_html(_fake_pngs(2), fps=20.0, width=100, height=100)
    assert f"}}, {int(1000.0 / 20.0)});" in html


def test_mesh_player_slider_range_matches_frame_count():
    html = build_mesh_player_html(_fake_pngs(7), fps=30.0, width=100, height=100)
    assert 'max="6"' in html  # 0-indexed, so max = count - 1


def test_mesh_player_handles_empty_frame_list_without_error():
    html = build_mesh_player_html([], fps=30.0, width=100, height=100)
    assert "const frames = []" in html
    assert 'max="0"' in html


# ---------------------------------------------------------------------------
# build_mesh_player_threejs (interactive mouse-orbit viewer, SMPL-X fit tab)
# ---------------------------------------------------------------------------
def _cube_sequence(t=4):
    """A tiny animated "mesh" (a cube translating upward each frame) — real
    SMPL-X geometry isn't needed to test the HTML/JS structure this
    function produces."""
    base = np.array([
        [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0], [-0.2, 0.2, 0.0],
        [-0.2, -0.2, 0.4], [0.2, -0.2, 0.4], [0.2, 0.2, 0.4], [-0.2, 0.2, 0.4],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    verts = np.stack([base + [0.0, 0.0, 0.1 * i] for i in range(t)])
    return verts, faces


def test_threejs_player_loads_three_and_orbit_controls():
    verts, faces = _cube_sequence()
    html = build_mesh_player_threejs(verts, faces, fps=30.0)
    assert "three.min.js" in html
    assert "OrbitControls.js" in html
    assert "new THREE.OrbitControls" in html


def test_threejs_player_has_static_ground_plane():
    verts, faces = _cube_sequence()
    html = build_mesh_player_threejs(verts, faces, fps=30.0)
    assert "PlaneGeometry" in html
    assert "GridHelper" in html
    # The ground must be positioned once from precomputed bounds, not
    # re-derived per frame — it should never move as frames advance.
    assert "ground.position.set" in html


def test_threejs_player_has_playback_and_view_controls():
    verts, faces = _cube_sequence()
    html = build_mesh_player_threejs(verts, faces, fps=24.0, width=300, height=300)
    assert 'id="tjs-play"' in html
    assert 'id="tjs-pause"' in html
    assert 'id="tjs-slider"' in html
    assert 'id="tjs-reset"' in html
    assert 'id="tjs-full"' in html
    assert "requestFullscreen" in html


def test_threejs_player_embeds_all_frames_and_faces():
    t = 5
    verts, faces = _cube_sequence(t=t)
    html = build_mesh_player_threejs(verts, faces, fps=30.0)
    assert f"const nFrames = {t};" in html
    assert f"const nVerts = {verts.shape[1]};" in html
    # The base64-embedded vertex buffer must actually be present and decode
    # back to the right total element count (t * n_verts * 3 float32s).
    start = html.index('b64ToBuffer("') + len('b64ToBuffer("')
    end = html.index('"', start)
    decoded = base64.b64decode(html[start:end])
    assert len(decoded) == t * verts.shape[1] * 3 * 4  # float32 = 4 bytes


def test_threejs_player_converts_z_up_to_y_up_by_default():
    """Vertices are Z-up in this project's convention; three.js is Y-up.
    The embedded buffer must reflect the rotated (y-up) coordinates, not
    the raw z-up input — checked by decoding the first frame back out."""
    verts, faces = _cube_sequence(t=1)
    html = build_mesh_player_threejs(verts, faces, fps=30.0, up_axis="z")
    start = html.index('b64ToBuffer("') + len('b64ToBuffer("')
    end = html.index('"', start)
    decoded = np.frombuffer(base64.b64decode(html[start:end]), dtype=np.float32).reshape(-1, 3)
    # Original z-up cube spans z in [0.0, 0.4] and y in [-0.2, 0.2]; after
    # rotating to y-up, that span must show up on the new y axis instead.
    assert decoded[:, 1].max() - decoded[:, 1].min() > 0.35
    assert decoded[:, 2].max() - decoded[:, 2].min() < 0.41


def test_threejs_player_rejects_bad_up_axis():
    verts, faces = _cube_sequence()
    with pytest.raises(ValueError):
        build_mesh_player_threejs(verts, faces, fps=30.0, up_axis="x")


def test_threejs_player_handles_empty_sequence_without_error():
    verts = np.zeros((0, 8, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    html = build_mesh_player_threejs(verts, faces, fps=30.0)
    assert "const nFrames = 0;" in html
    assert 'max="0"' in html
