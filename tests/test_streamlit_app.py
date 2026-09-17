"""Streamlit UI tests.

`AppTest` (Streamlit's official script-testing harness) runs the actual app
script in a simulated session and lets us assert on the rendered element
tree and catch unhandled exceptions — it does not, as of this Streamlit
version, support simulating a `st.file_uploader` drag-and-drop, so the
upload-driven rendering path is exercised via `AppTest.from_function`
wrapping the same `render_raw_tab`/`render_fit_tab` functions the real app
uses, fed with a real uploaded file's bytes. Between this and
tests/test_app_pipeline.py (which exercises `process_upload` itself
end-to-end against real files), the whole app is covered without a browser.
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

REAL_NPZ = "E:/github_clone/GPSM-1/data/0013_knocking1_poses.npz"
REAL_C3D = "E:/github_clone/GPSM-1/data/01_01.c3d"


def test_app_loads_with_no_upload():
    """Smoke test: the app must start cleanly, show the model status and
    the "please upload" prompt, and raise no exception — with no model
    file needed, since this only needs to reach the model-not-found branch
    or the success branch, either way without crashing."""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception
    # Either the "no model" error or the "model loaded" success must show —
    # never a silent blank page.
    all_text = " ".join(e.value for e in at.error) + " ".join(e.value for e in at.success) \
        + " ".join(e.value for e in at.info)
    assert ("SMPL-X model" in all_text) or ("Upload a file" in all_text)


def _render_uploaded_file(file_bytes: bytes, filename: str, max_fit_frames: int):
    """Runs inside AppTest.from_function — mirrors app.streamlit_app.main()
    but with the upload hard-coded instead of coming from a widget, since
    AppTest cannot simulate a real file_uploader drag-and-drop."""
    import streamlit as st

    from app.pipeline import process_upload
    from app.streamlit_app import get_body_model, render_fit_tab, render_raw_tab

    body = get_body_model()
    raw, fit = process_upload(file_bytes, filename, body, max_fit_frames=max_fit_frames)

    tab_raw, tab_fit = st.tabs(["1. Raw Mocap", "2. SMPL-X Fit"])
    with tab_raw:
        render_raw_tab(raw)
    with tab_fit:
        render_fit_tab(fit, body)


@pytest.mark.skipif(not os.path.isfile(REAL_NPZ), reason="real npz sample not found")
def test_render_tabs_for_real_npz_upload(smplx_model_path):
    with open(REAL_NPZ, "rb") as f:
        data = f.read()
    at = AppTest.from_function(
        _render_uploaded_file, default_timeout=120,
        kwargs=dict(file_bytes=data, filename=os.path.basename(REAL_NPZ), max_fit_frames=8),
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.tabs) == 2
    # The fit tab should report a verdict metric and the "Tier 0" note.
    all_captions = " ".join(c.value for c in at.caption)
    assert "Tier 0" in all_captions
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Fit quality" in all_markdown


@pytest.mark.skipif(not os.path.isfile(REAL_C3D), reason="real c3d sample not found")
def test_render_tabs_for_real_c3d_upload(smplx_model_path):
    with open(REAL_C3D, "rb") as f:
        data = f.read()
    at = AppTest.from_function(
        _render_uploaded_file, default_timeout=120,
        kwargs=dict(file_bytes=data, filename=os.path.basename(REAL_C3D), max_fit_frames=8),
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    all_captions = " ".join(c.value for c in at.caption)
    assert "VICON_50" in all_captions or "markers" in all_captions


@pytest.mark.skipif(not os.path.isfile(REAL_NPZ), reason="real npz sample not found")
def test_raw_tab_playback_is_client_side_not_server_driven(smplx_model_path):
    """Regression test for the reported bug (clicking Play made the page
    unscrollable, and its "view fullscreen" control unresponsive).

    Two earlier attempts were tried and superseded:
      1. A blocking `for` loop over every frame inside one script
         execution — kept the backend in a permanently "still running"
         state for the whole clip's duration.
      2. One frame per `st.rerun()`-triggered script execution — each
         individual step was fast and bounded, but the backend was still
         continuously busy back-to-back for the entire playthrough, which
         is what turned out to still be fighting the browser's scroll and
         click handling.

    The fix removes Python from the playback loop entirely: this test
    asserts the *rendering path itself*, not just timing, proves that —
    there must be no `frame_selector`/`advance_playback`-style
    session-state machinery left (no `raw_playing`/`raw_frame` keys ever
    appear), and no `st.button` Play/Pause widgets in the raw tab, because
    playback for it is now baked entirely into one Plotly figure's native
    animation (see tests/test_client_side_playback.py for direct
    assertions on that figure's `frames`/`updatemenus`/`sliders`).
    """
    with open(REAL_NPZ, "rb") as f:
        data = f.read()
    at = AppTest.from_function(
        _render_uploaded_file, default_timeout=30,
        kwargs=dict(file_bytes=data, filename=os.path.basename(REAL_NPZ), max_fit_frames=8),
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    assert "raw_playing" not in at.session_state
    assert "raw_frame" not in at.session_state
    button_keys = {b.key for b in at.button}
    assert "raw_play_btn" not in button_keys
    assert "raw_pause_btn" not in button_keys


@pytest.mark.skipif(not os.path.isfile(REAL_NPZ), reason="real npz sample not found")
def test_fit_tab_playback_is_client_side_not_server_driven(smplx_model_path):
    """Same guarantee for the SMPL-X Fit tab — the heavier of the two
    paths (a real pyrender mesh render per frame), where a server-driven
    animation loop would be the most costly to leave running unattended.
    Playback here is a pre-rendered image list handed to a plain HTML/JS
    player (see tests/test_client_side_playback.py for direct assertions
    on that player's generated markup)."""
    with open(REAL_NPZ, "rb") as f:
        data = f.read()
    at = AppTest.from_function(
        _render_uploaded_file, default_timeout=60,
        kwargs=dict(file_bytes=data, filename=os.path.basename(REAL_NPZ), max_fit_frames=5),
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    assert "fit_playing" not in at.session_state
    assert "fit_frame" not in at.session_state
    button_keys = {b.key for b in at.button}
    assert "fit_play_btn" not in button_keys
    assert "fit_pause_btn" not in button_keys
