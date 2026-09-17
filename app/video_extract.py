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

from src.smplx.layouts.builtin import MEDIAPIPE_33_POINT_NAMES

# MediaPipe's own landmark ordering matches MEDIAPIPE_33_POINT_NAMES exactly
# (verified: mp.solutions.pose.PoseLandmark enum order) — no reordering needed.


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
    progress_callback=None,
) -> RawVideoCapture:
    """Run MediaPipe Pose over a video file, frame by frame.

    `max_frames`: hard cap so a long upload can't hang the app indefinitely;
    frames beyond the cap are simply not read. `progress_callback(i, n)` is
    called after each processed frame, for a UI progress bar.
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
        model_complexity=1,
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
