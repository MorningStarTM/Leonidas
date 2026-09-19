"""Extract real 3D landmarks from a video file via MediaPipe Pose.

MediaPipe's `pose_world_landmarks` are a metric-scale, hip-centered 3D
estimate (not pixel coordinates) — real 3D mocap, produced by a general
public, freely-available library, not a placeholder. This lets video
input feed directly into the Tier-1 solver (src/smplx/adapters/skeleton3d)
via the existing `mediapipe_33` layout, with no need for a heavier
whole-body regressor checkpoint (see src/smplx/regressor.py's honest
limitation notes for why that path is stubbed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.smplx import ops as motion_ops
from src.smplx.layouts.builtin import MEDIAPIPE_33_POINT_NAMES

# MediaPipe's own landmark ordering matches MEDIAPIPE_33_POINT_NAMES exactly
# (verified: mp.solutions.pose.PoseLandmark enum order) — no reordering needed.


def _mediapipe_world_to_zup(points: np.ndarray) -> np.ndarray:
    """Convert MediaPipe's native `pose_world_landmarks` axis convention to
    this project's standard Z-up world frame.

    This was a real, confirmed bug, not a guess: this pipeline previously
    declared `up_axis="z"` for video unconditionally, which — because the
    *declared* axis already matched the *target* axis — made the whole
    up-axis conversion step a silent no-op, so MediaPipe's raw coordinates
    were passed straight through untouched.

    Measured directly against three different real videos (not synthetic
    data): the mean nose-to-ankle offset is completely dominated by Y in
    every case (-0.88, -1.23, -1.22), an order of magnitude larger than X
    or Z, and always *negative* — meaning the head sits at a lower Y value
    than the feet. MediaPipe's world landmarks use the same Y-down
    convention as raw image pixel coordinates (rescaled to metric,
    hip-centered), not a Y-up or Z-up convention.

    The correction is a 180-degree rotation about X — (x, y, z) -> (x, -y,
    -z) — deliberately not a single-axis sign flip. Negating only Y would
    change the frame's handedness (determinant -1), which is a mirror
    reflection: it would still put "up" in the right place, but it would
    also swap the body's anatomical left and right, twisting every
    rotation the wrong way — almost certainly the deeper cause behind the
    reported "extracted points didn't match the real actions" symptom, not
    just the axis being visually sideways. Rotating 180 degrees about X
    fixes the up direction while provably preserving handedness (rotations
    always have determinant +1), then the result — now a standard Y-up
    frame — is passed through this project's existing, already-tested
    Y-up -> Z-up conversion (`ops.convert_up_axis`) to match every other
    adapter's convention.
    """
    y_up = points.copy()
    y_up[..., 1] *= -1.0
    y_up[..., 2] *= -1.0
    return motion_ops.convert_up_axis(y_up, "y", "z")


def _interpolate_short_gaps(points: np.ndarray, conf: np.ndarray, max_gap: int = 3):
    """Linearly bridge short runs of fully-lost frames (<= max_gap
    consecutive frames with no MediaPipe detection at all) when bounded by
    a good detection on both sides. Longer runs, and gaps touching the very
    start or end of the clip (nothing to interpolate from), are left
    untouched.

    This is a deliberate middle ground, not a relaxation of the "never
    silently guess" rule used throughout this project: a one- or
    two-frame tracking hiccup bounded by two real detections is safe to
    bridge (the true pose almost certainly moved smoothly between them),
    whereas a longer gap is not assumed away — it stays marked
    unobserved, same as before. Ported from the same technique in a
    sibling project's offline landmark-extraction pipeline
    (labeling-pipeline-v1/backend/scripts/_features.py::interpolate_nan_gaps),
    simplified here because MediaPipe's failure mode is whole-frame (every
    landmark is lost together), not per-landmark, so the interpolation can
    run per-frame directly instead of per-channel.

    Interpolated frames get a conservative confidence — the lower of the
    two bounding frames' per-landmark confidence, halved — never a value
    implying they were actually measured.
    """
    points = points.copy()
    conf = conf.copy()
    t = points.shape[0]
    is_lost = ~np.isfinite(points).all(axis=(1, 2))  # (T,) whole-frame lost

    i = 0
    while i < t:
        if not is_lost[i]:
            i += 1
            continue
        start = i
        while i < t and is_lost[i]:
            i += 1
        end = i  # exclusive
        length = end - start
        if start == 0 or end == t or length > max_gap:
            continue  # leading/trailing, or too long: leave as unobserved

        left, right = start - 1, end
        for f in range(start, end):
            alpha = (f - left) / (right - left)
            points[f] = (1 - alpha) * points[left] + alpha * points[right]
            conf[f] = np.minimum(conf[left], conf[right]) * 0.5

    return points, conf


@dataclass
class RawVideoCapture:
    points_m: np.ndarray       # (T, 33, 3) world-space landmarks, meters
    conf: np.ndarray           # (T, 33) visibility, 0..1
    fps: float
    frame_size: "tuple[int, int]"
    preview_frames: List[np.ndarray]  # a handful of RGB frames, for the raw-tab display


def extract_pose_from_video(
    video_path: str,
    max_frames: Optional[int] = 300,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 2,
    max_interpolated_gap: int = 3,
    progress_callback=None,
) -> RawVideoCapture:
    """Run MediaPipe Pose over a video file, frame by frame.

    `max_frames`: hard cap so a long upload can't hang the app indefinitely;
    frames beyond the cap are simply not read. `progress_callback(i, n)` is
    called after each processed frame, for a UI progress bar.

    `model_complexity`: 2 (MediaPipe's heaviest, most accurate variant) by
    default — confirmed via a sibling project's own offline landmark
    extraction pipeline (labeling-pipeline-v1) to be the setting actually
    used there, versus this project's previous default of 1.

    `max_interpolated_gap`: short runs of fully-lost frames (a MediaPipe
    tracking hiccup) up to this many frames long are linearly bridged when
    bounded by a good detection on both sides — see
    `_interpolate_short_gaps`. Set to 0 to disable and mark every lost
    frame unobserved, as before.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    n_target = min(total, max_frames) if (total and max_frames) else (max_frames or total)

    points_list, conf_list, preview_frames = [], [], []

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_world_landmarks is not None:
                lm = result.pose_world_landmarks.landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
                vis = np.array([p.visibility for p in lm], dtype=np.float64)
            else:
                pts = np.full((33, 3), np.nan)
                vis = np.zeros(33)

            points_list.append(pts)
            conf_list.append(vis)
            if frame_idx % max(1, (n_target or 30) // 6) == 0 and len(preview_frames) < 6:
                preview_frames.append(frame_rgb)

            frame_idx += 1
            if progress_callback is not None and n_target:
                progress_callback(frame_idx, n_target)
    finally:
        pose.close()
        cap.release()

    if not points_list:
        raise ValueError("No frames could be read from this video file.")

    points = np.stack(points_list, axis=0)
    conf = np.stack(conf_list, axis=0)

    # Correct MediaPipe's native Y-down convention to this project's
    # standard Z-up world frame — see _mediapipe_world_to_zup's docstring.
    # Done first, before gap interpolation or zeroing lost frames: it's a
    # pure per-point rotation, safe to apply uniformly (NaN stays NaN).
    points = _mediapipe_world_to_zup(points)

    # Bridge short tracking hiccups first (see _interpolate_short_gaps) —
    # only frames still lost afterward (long gaps, or ones touching the
    # very start/end of the clip) fall through to the next step.
    if max_interpolated_gap > 0:
        points, conf = _interpolate_short_gaps(points, conf, max_gap=max_interpolated_gap)

    # A frame where MediaPipe lost tracking entirely (all-NaN) must not be
    # silently treated as "at the origin" — zero it out AND zero its
    # confidence, consistent with the "never silently zero-fill an
    # unobserved point" rule used throughout this pipeline.
    lost = ~np.isfinite(points).all(axis=-1)
    points[lost] = 0.0
    conf[lost] = 0.0

    return RawVideoCapture(
        points_m=points, conf=conf, fps=float(fps),
        frame_size=(width, height), preview_frames=preview_frames,
    )


POSE_CONNECTIONS_BY_NAME = None  # populated lazily to avoid importing mediapipe at module load


def get_mediapipe_skeleton_edges() -> List[tuple]:
    """(index, index) bone pairs in MEDIAPIPE_33_POINT_NAMES order, from
    MediaPipe's own connection graph."""
    import mediapipe as mp

    name_to_idx = {name: i for i, name in enumerate(MEDIAPIPE_33_POINT_NAMES)}
    landmark_names = [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]
    edges = []
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        edges.append((name_to_idx[landmark_names[a]], name_to_idx[landmark_names[b]]))
    return edges
