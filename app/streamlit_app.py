"""Leonidas mocap viewer — Streamlit app.

Upload a mocap file (.npz SMPL/SMPL-H/SMPL-X params, .c3d optical markers, .bvh skeleton animation,
or a video) and see it two ways:

  1. Raw Mocap  — the data exactly as the source format measured it
     (parametric skeleton, marker cloud, or MediaPipe 3D landmarks).
  2. SMPL-X Fit — the same clip unified onto the real SMPL-X body model
     and rendered in an interactive, mouse-orbitable 3D viewport (three.js)
     with a static ground plane.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Allow `streamlit run app/streamlit_app.py` to find the `src`/`app`
# packages regardless of the invoker's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from app.model_bootstrap import ensure_smplx_model_downloaded
from app.pipeline import UnsupportedFileError, process_upload
from app.render3d import build_mesh_player_threejs, compute_axis_bounds, plot_points_animation
from src.smplx.config import SMPLXModelNotFoundError, find_smplx_model
from src.smplx.fitting.body import BodyParams, SMPLXBody

st.set_page_config(page_title="Leonidas Mocap Viewer", layout="wide")


@st.cache_resource(show_spinner="Loading SMPL-X body model...")
def get_body_model():
    return SMPLXBody()


@st.cache_data(show_spinner="Processing upload...")
def cached_process_upload(_body, file_bytes: bytes, filename: str, max_fit_frames: int, max_extract_frames: int):
    # `_body` (leading underscore) is excluded from Streamlit's cache-key
    # hashing — it's a heavyweight torch module, not something we want (or
    # can cheaply) hash; the remaining args fully determine the result
    # otherwise, so caching stays correct.
    return process_upload(file_bytes, filename, _body,
                           max_fit_frames=max_fit_frames, max_extract_frames=max_extract_frames)


@st.cache_data(show_spinner="Preparing SMPL-X mesh viewer...")
def cached_mesh_player_html(vertices_bytes, faces_bytes, shape, fps, width, height):
    """Build the interactive three.js viewer's HTML once per (clip, size).
    Unlike the earlier pyrender-based player, no server-side rendering
    happens here at all — the vertex data is just serialized for the
    browser's own WebGL renderer to draw and let the user orbit freely."""
    all_vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(shape)
    faces = np.frombuffer(faces_bytes, dtype=np.int64).reshape(-1, 3)
    return build_mesh_player_threejs(all_vertices, faces, fps=fps, width=width, height=height)


def _plotly_chart_full_width(fig, key):
    # Newer Streamlit replaced `use_container_width` with `width="stretch"`;
    # older versions only know the former. Pick by signature, not by version.
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _show_html(html: str, height: int):
    # `st.components.v1.html` is deprecated in favor of `st.iframe`, which
    # takes a raw HTML string directly. Fall back for older Streamlit.
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def render_raw_tab(raw):
    st.subheader("Raw Mocap — as measured by the source format")
    for note in raw.notes:
        st.caption(f"• {note}")

    bounds = compute_axis_bounds(raw.points)
    fig = plot_points_animation(
        raw.points, conf_seq=raw.conf, edges=raw.edges, labels=raw.labels,
        bounds=bounds, fps=raw.fps, title=f"{raw.kind}",
    )
    # Every frame is baked into this one figure as a native Plotly
    # animation (frames + Play/Pause + a scrub slider) — the browser
    # animates it entirely on its own from here; Streamlit's Python script
    # does not run again while it plays, which is what keeps the page
    # scrollable and its native "view fullscreen" control responsive
    # during playback (see plot_points_animation's docstring).
    _plotly_chart_full_width(fig, key=f"raw_plot_{id(raw)}")

    n_unobserved = int((raw.conf <= 0).sum())
    if n_unobserved:
        st.caption(f"Red points are unobserved ({n_unobserved} point-frames) — shown, not hidden, "
                   "so gaps in the source data stay visible.")


def render_fit_tab(fit, body):
    st.subheader("SMPL-X Fit — unified onto the real body model")
    if fit.motion is None:
        st.warning(fit.unavailable_reason or "No SMPL-X fit is available for this clip.")
        return

    quality = fit.quality
    verdict_color = {"GOOD": "green", "DEGRADED": "orange", "REJECT": "red"}[quality.verdict]
    st.markdown(f"**Fit quality: :{verdict_color}[{quality.verdict}]**")
    cols = st.columns(4)
    cols[0].metric("Observed fraction", f"{quality.observed_fraction * 100:.0f}%")
    if not np.isnan(quality.joint_rmse_mm):
        cols[1].metric("Joint RMSE", f"{quality.joint_rmse_mm:.1f} mm")
    else:
        cols[1].metric("Joint RMSE", "n/a (no fitting error)")
    cols[2].metric("Accel spikes", quality.accel_spike_count)
    cols[3].metric("Frames", fit.motion.num_frames)
    for note in fit.notes:
        st.caption(f"• {note}")
    if fit.motion.part_mask[:, 2].max() == 0 or fit.motion.part_mask[:, 3].max() == 0:
        st.caption("Hands are shown at rest — not enough correspondences in this input "
                   "to fit real hand articulation (see src/smplx/README.md).")

    with st.expander("Render settings"):
        c1, c2 = st.columns(2)
        width = c1.select_slider("Viewport width", options=[480, 560, 640, 720, 860], value=640, key="mesh_width")
        height = c2.select_slider("Viewport height", options=[360, 420, 480, 540, 640], value=480, key="mesh_height")

    import torch
    t = fit.motion.num_frames
    params = BodyParams(
        betas=torch.tensor(fit.motion.betas), global_orient=torch.tensor(fit.motion.global_orient),
        body_pose=torch.tensor(fit.motion.body_pose), left_hand_pose=torch.tensor(fit.motion.left_hand_pose),
        right_hand_pose=torch.tensor(fit.motion.right_hand_pose), transl=torch.tensor(fit.motion.trans),
    )
    with torch.no_grad():
        out = body.forward(params, return_verts=True)
    all_vertices = out.vertices.numpy()
    faces = body.layer.faces.astype(np.int64)

    # The mesh geometry itself (not a pre-rendered image) is handed to a
    # client-side three.js viewport: drag-to-orbit, scroll-to-zoom, a
    # static ground grid, and its own Play/Pause/scrub/Fullscreen controls
    # — see build_mesh_player_threejs's docstring for why this replaces
    # the earlier fixed-angle pyrender player.
    html = cached_mesh_player_html(
        all_vertices.tobytes(), faces.tobytes(), all_vertices.shape,
        float(fit.motion.fps), int(width), int(height),
    )
    _show_html(html, height=height + 110)


def main():
    st.title("Leonidas Mocap Viewer")
    st.caption("Upload mocap data (.npz SMPL/SMPL-H/SMPL-X params, .c3d optical markers, .bvh skeleton animation, "
               "or a video) to compare it raw against a unified SMPL-X fit.")

    try:
        with st.spinner("Checking for SMPL-X model..."):
            bootstrap_status = ensure_smplx_model_downloaded()
    except Exception as e:
        st.error(f"SMPL-X model download failed: {e}")
        st.stop()

    model_path = find_smplx_model()
    if model_path is None:
        st.error(
            "No SMPL-X model file found. Set the LEONIDAS_SMPLX_MODEL environment variable "
            "to a SMPLX_*.npz file (registration-gated, see https://smpl-x.is.tue.mpg.de) "
            "before running this app."
        )
        st.caption(f"Bootstrap: {bootstrap_status}")
        st.stop()
    st.sidebar.success(f"SMPL-X model: {model_path.name}")

    uploaded = st.sidebar.file_uploader(
        "Upload mocap data", type=["npz", "c3d", "bvh", "mp4", "avi", "mov", "mkv", "webm"],
    )
    st.sidebar.caption("The Raw Mocap tab always shows the complete clip — the setting below only "
                       "controls how much of it the SMPL-X Fit tab fits and renders as a mesh.")
    max_fit_frames = st.sidebar.slider(
        "Frames to fit / render (SMPL-X tab)", min_value=5, max_value=300, value=60, step=5,
        help="Fitting is an iterative optimization and mesh rendering runs once per frame, so both "
             "scale with frame count. Longer clips are evenly subsampled across their *entire* "
             "duration to stay within this budget — not truncated to just the opening seconds.",
    )
    max_extract_frames = st.sidebar.slider(
        "Video frames to extract", min_value=10, max_value=300, value=150, step=10,
        help="Only applies to video uploads: MediaPipe runs a neural network on every extracted "
             "frame, so this bounds extraction itself (the Raw Mocap tab shows whatever was "
             "extracted; the fit setting above then subsamples from within that).",
    )

    if uploaded is None:
        st.info("Upload a file from the sidebar to begin.")
        return

    body = get_body_model()
    file_bytes = uploaded.getvalue()

    try:
        with st.spinner(f"Processing {uploaded.name}..."):
            raw, fit = cached_process_upload(body, file_bytes, uploaded.name, max_fit_frames, max_extract_frames)
    except UnsupportedFileError as e:
        st.error(str(e))
        return
    except SMPLXModelNotFoundError as e:
        st.error(str(e))
        return

    tab_raw, tab_fit = st.tabs(["1. Raw Mocap", "2. SMPL-X Fit"])
    with tab_raw:
        render_raw_tab(raw)
    with tab_fit:
        render_fit_tab(fit, body)


if __name__ == "__main__":
    main()
