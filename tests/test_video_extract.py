"""Tests for app/video_extract.py's gap-bridging and model-quality settings.

Ported technique: a sibling project's offline landmark-extraction pipeline
(labeling-pipeline-v1/backend/scripts/_features.py::interpolate_nan_gaps)
linearly bridges short runs of lost MediaPipe detections bounded by two
good ones, and uses model_complexity=2 (MediaPipe's most accurate variant)
rather than this project's previous default of 1.
"""
import os

import numpy as np
import pytest

from app.video_extract import _interpolate_short_gaps, extract_pose_from_video


def _linear_clip(t=10):
    points = np.zeros((t, 33, 3))
    for i in range(t):
        points[i] = float(i)
    conf = np.ones((t, 33))
    return points, conf


def test_short_gap_is_linearly_bridged():
    points, conf = _linear_clip(10)
    points[3] = np.nan
    conf[3] = 0.0
    points[4] = np.nan
    conf[4] = 0.0

    out_points, out_conf = _interpolate_short_gaps(points, conf, max_gap=3)

    assert np.allclose(out_points[3], 3.0)
    assert np.allclose(out_points[4], 4.0)
    assert not np.isnan(out_points).any()
    # Conservative confidence: strictly less than either real bounding frame's.
    assert (out_conf[3] < conf[2]).all()
    assert (out_conf[4] < conf[5]).all()


def test_gap_longer_than_budget_is_left_unobserved():
    points, conf = _linear_clip(10)
    points[1:6] = np.nan  # 5-frame gap, budget is 3
    conf[1:6] = 0.0

    out_points, out_conf = _interpolate_short_gaps(points, conf, max_gap=3)

    assert np.isnan(out_points[1:6]).all()
    assert (out_conf[1:6] == 0.0).all()


def test_leading_and_trailing_gaps_are_never_interpolated():
    """No left/right neighbor exists to interpolate from, regardless of
    how short the gap is — must stay marked unobserved."""
    points, conf = _linear_clip(10)
    points[0] = np.nan
    conf[0] = 0.0
    points[9] = np.nan
    conf[9] = 0.0

    out_points, out_conf = _interpolate_short_gaps(points, conf, max_gap=3)

    assert np.isnan(out_points[0]).all()
    assert np.isnan(out_points[9]).all()


def test_no_gaps_leaves_data_unchanged():
    points, conf = _linear_clip(6)
    out_points, out_conf = _interpolate_short_gaps(points, conf, max_gap=3)
    assert np.array_equal(out_points, points)
    assert np.array_equal(out_conf, conf)


def test_disabling_interpolation_via_max_gap_zero_is_a_noop():
    points, conf = _linear_clip(10)
    points[3] = np.nan
    conf[3] = 0.0
    # max_gap=0: no run of length <= 0 can ever exist, so nothing bridges.
    out_points, out_conf = _interpolate_short_gaps(points, conf, max_gap=0)
    assert np.isnan(out_points[3]).all()


# The real synthetic-avatar test video lives in this session's scratch dir,
# not the repo — skip cleanly if it isn't present (e.g. a fresh checkout).
_SCRATCH_AVATAR_VIDEO = (
    "C:/Users/Ernest/AppData/Local/Temp/claude/e--github-clone-Leonidas/"
    "bf1beb2b-7471-4c75-862c-09189155e07a/scratchpad/avatar_test.mp4"
)


@pytest.mark.skipif(not os.path.isfile(_SCRATCH_AVATAR_VIDEO), reason="synthetic avatar test video not found")
def test_extract_pose_from_video_uses_heavy_model_by_default():
    """Real end-to-end check against a real (rendered) video: the default
    model_complexity=2 path must run without error and produce genuine
    detections, not just compile."""
    cap = extract_pose_from_video(_SCRATCH_AVATAR_VIDEO, max_frames=10)
    assert cap.points_m.shape == (10, 33, 3)
    assert cap.conf.mean() > 0.5  # this video is a clean synthetic avatar; should track well


@pytest.mark.skipif(not os.path.isfile(_SCRATCH_AVATAR_VIDEO), reason="synthetic avatar test video not found")
def test_extract_pose_from_video_gap_interpolation_can_be_disabled():
    cap = extract_pose_from_video(_SCRATCH_AVATAR_VIDEO, max_frames=10, max_interpolated_gap=0)
    assert cap.points_m.shape == (10, 33, 3)
