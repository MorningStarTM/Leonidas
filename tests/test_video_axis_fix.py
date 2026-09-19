"""Regression tests for the reported bug: video-extracted mocap did not
match the real actions in the source video, and the raw simulation's
X/Y/Z orientation was wrong.

Root cause (confirmed against three independent real videos, not
synthetic data): MediaPipe's `pose_world_landmarks` use a Y-down
convention (the same as raw image pixel coordinates, rescaled to meters),
not the Z-up convention every other adapter in this project uses. The
pipeline previously declared `up_axis="z"` for video unconditionally,
which silently became a no-op (declared already equalled the target), so
MediaPipe's raw, Y-down coordinates passed straight through untouched —
this is exactly what made the raw skeleton view look rotated onto its
side, and fed the same wrongly-oriented data into the fitting solver.

The fix (`_mediapipe_world_to_zup`) applies a 180-degree rotation about X
— not a single-axis sign flip, which would silently mirror the body's
left and right sides (a determinant -1 reflection) instead of correcting
only the "up" direction.
"""
import os

import numpy as np
import pytest

from app.video_extract import _mediapipe_world_to_zup, extract_pose_from_video
from src.smplx.layouts.builtin import MEDIAPIPE_33_POINT_NAMES

SQUAT_VIDEO = "C:/Users/Ernest/Downloads/squat_correct_men_v3.mp4"
requires_squat_video = pytest.mark.skipif(not os.path.isfile(SQUAT_VIDEO), reason="real squat video not found")

_NAME_IDX = {n: i for i, n in enumerate(MEDIAPIPE_33_POINT_NAMES)}


def test_correction_is_a_proper_rotation_not_a_mirror():
    """The fix must preserve handedness (determinant +1) — flipping only Y
    would put "up" in the right place but silently mirror left/right,
    twisting every joint rotation the wrong way. A 180-degree rotation
    about X must not do that: applying it to a right-handed orthonormal
    basis must produce another right-handed basis."""
    basis = np.eye(3).reshape(1, 3, 3)  # one "frame" of 3 basis vectors as points
    corrected = _mediapipe_world_to_zup(basis[0])
    det = np.linalg.det(corrected)
    assert abs(det - 1.0) < 1e-9, f"correction changed handedness (det={det}), it must be a proper rotation"


def test_correction_maps_known_mediapipe_convention_to_zup():
    """Directly verify the transform on synthetic points shaped like
    MediaPipe's own documented/measured convention: head at negative Y
    (up, in MediaPipe's frame), feet at positive Y (down)."""
    # (x, y, z): head near origin-ish negative Y, feet at positive Y.
    points = np.array([
        [0.0, -0.8, 0.1],   # "head", mediapipe-native: high up = very negative Y
        [0.0, 0.8, 0.1],    # "feet", mediapipe-native: down = positive Y
    ])
    out = _mediapipe_world_to_zup(points)
    head_z, feet_z = out[0, 2], out[1, 2]
    assert head_z > feet_z, "after correction, head must be at a HIGHER Z than feet"


@requires_squat_video
def test_real_video_head_ends_up_above_feet_on_z():
    """End-to-end, against a real video (not synthetic): the nose must sit
    at a higher Z than the ankles after extraction, confirming the fix is
    wired into the actual extraction path, not just the standalone
    transform function."""
    cap = extract_pose_from_video(SQUAT_VIDEO, max_frames=60)
    nose_z = cap.points_m[:, _NAME_IDX["nose"], 2]
    ankle_z = (cap.points_m[:, _NAME_IDX["left_ankle"], 2]
               + cap.points_m[:, _NAME_IDX["right_ankle"], 2]) / 2
    diff = float(np.mean(nose_z - ankle_z))
    assert diff > 0.5, f"nose should sit well above ankles on Z after the fix; got diff={diff:.3f}"


@requires_squat_video
def test_real_video_up_axis_dominates_over_horizontal_axes():
    """The vertical (Z) separation between head and feet should be
    substantially larger than the horizontal (X) separation — sanity
    check that Z, not X, is now carrying the "up" signal."""
    cap = extract_pose_from_video(SQUAT_VIDEO, max_frames=60)
    nose = cap.points_m[:, _NAME_IDX["nose"], :]
    ankle = (cap.points_m[:, _NAME_IDX["left_ankle"], :] + cap.points_m[:, _NAME_IDX["right_ankle"], :]) / 2
    diff = np.abs((nose - ankle).mean(axis=0))
    assert diff[2] > diff[0] * 2, f"Z separation should dominate X separation; got diff={diff}"


@requires_squat_video
def test_real_video_captures_genuine_squat_cycle_motion():
    """The actual point of the fix: extracted mocap must reflect the real
    action in the video. A squat produces a clear, physically sensible
    oscillation in hip-to-ankle height (large when standing, small at the
    bottom of the squat) — verify that oscillation is present and has a
    plausible magnitude, not a flat or nonsensical signal."""
    cap = extract_pose_from_video(SQUAT_VIDEO, max_frames=200)
    hip = (cap.points_m[:, _NAME_IDX["left_hip"], :] + cap.points_m[:, _NAME_IDX["right_hip"], :]) / 2
    ankle = (cap.points_m[:, _NAME_IDX["left_ankle"], :] + cap.points_m[:, _NAME_IDX["right_ankle"], :]) / 2
    hip_to_ankle_height = ankle[:, 2] - hip[:, 2]  # negative; magnitude = standing height

    height_range = float(hip_to_ankle_height.max() - hip_to_ankle_height.min())
    assert height_range > 0.2, (
        f"expected a clear standing/squatting height oscillation (>0.2m); got range={height_range:.3f}m"
    )
    # Plausible human-scale bounds: never more than ~1m of hip-to-ankle
    # height (that would indicate a broken/ridiculous scale), and the
    # oscillation shouldn't exceed the maximum standing height itself.
    assert -1.0 < hip_to_ankle_height.min() < 0
    assert height_range < abs(hip_to_ankle_height.min()) + 0.1
