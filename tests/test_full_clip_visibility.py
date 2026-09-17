"""Regression tests for the reported bug: the Raw Mocap tab silently
truncated real captures longer than a fixed frame cap (200, or the UI's
default of 30), so a multi-second capture's "full simulation" was never
visible — only its opening fraction of a second.

The fix decouples two previously-conflated concerns in app/pipeline.py:
  - The Raw Mocap tab always shows the *complete* clip (cheap: no fitting,
    no per-frame optimization).
  - Only the SMPL-X Fit tab's frame budget (fitting + pyrender mesh
    rendering, which do have real per-frame cost) is capped, and that cap
    is applied by evenly subsampling across the *whole* clip rather than
    truncating to its start.
"""
import os

import numpy as np
import pytest

from app.pipeline import process_upload

REAL_2017_DIR = "C:/Users/Ernest/Downloads/MoCap/2017-06-22"
_sample_2017 = os.path.join(REAL_2017_DIR, "00006_raw.c3d")
requires_2017_sample = pytest.mark.skipif(
    not os.path.isfile(_sample_2017), reason="referenced 2017-06-22 sample dataset not found"
)


@requires_2017_sample
def test_long_real_capture_fully_visible_in_raw_tab(smplx_body):
    """The exact scenario reported: a real file from this dataset runs
    well over 200 frames — the old fixed cap (200, and a UI default of
    30) would have silently truncated the raw view. It must not anymore,
    regardless of what the fit-tab frame budget is set to."""
    from src.smplx.adapters.markers import load_c3d

    cap = load_c3d(_sample_2017)
    assert cap.points_m.shape[0] > 200, "test file must actually exceed the old cap to be meaningful"

    with open(_sample_2017, "rb") as f:
        data = f.read()

    # Even with a tiny fit budget, the raw view must show everything.
    raw, fit = process_upload(data, os.path.basename(_sample_2017), smplx_body, max_fit_frames=15)
    assert raw.points.shape[0] == cap.points_m.shape[0]
    assert not np.isnan(raw.points).any()
    assert fit.motion is None or fit.motion.num_frames == 15


@requires_2017_sample
def test_raw_tab_frame_count_independent_of_fit_budget(smplx_body):
    """Changing the fit-tab frame budget must never change how many
    frames the raw tab shows — the two are unrelated resource budgets,
    not one shared cap (the root cause of the reported bug)."""
    with open(_sample_2017, "rb") as f:
        data = f.read()

    raw_small, _ = process_upload(data, os.path.basename(_sample_2017), smplx_body, max_fit_frames=5)
    raw_large, _ = process_upload(data, os.path.basename(_sample_2017), smplx_body, max_fit_frames=250)
    assert raw_small.points.shape[0] == raw_large.points.shape[0]
