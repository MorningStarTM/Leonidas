"""Orchestration: uploaded file bytes -> (raw capture, fitted CanonicalMotion).

Kept separate from the Streamlit UI script so it can be exercised by plain
pytest without a running Streamlit server.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.smplx.adapters.bvh import match_bvh_layout, load_bvh
from src.smplx.adapters.markers import MIN_MATCHED_MARKERS, build_matched_layout, load_c3d
from src.smplx.adapters.params import UnsupportedFormatError, ingest_params
from src.smplx.fitting.body import BodyParams, SMPLXBody
from src.smplx.fitting.solver import (
    SolverConfig, StageSpec, fit_observation, scale_temporal_weights_for_subsampling,
)
from src.smplx.layouts.builtin import MEDIAPIPE_33
from src.smplx.quality import score_canonical_motion
from src.smplx.schema import CanonicalMotion, FitQuality, Observation

SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# The default solver schedule (src/smplx/fitting/solver.py::DEFAULT_STAGES)
# is tuned against real C3D marker data and left alone here to avoid
# regressing that already-validated tier. MediaPipe's mediapipe_33 layout
# is a different kind of input — every correspondence already carries a
# reduced structural weight (surface points, not joint centers — see
# layouts/builtin.py), and real video adds real occlusion (a barbell, a
# rack) that marker/skeleton data doesn't have. Measured directly against
# a real gym squat video: this schedule (more iterations, a tighter robust-
# loss radius, less pose-prior pull once real data exists) reduced joint
# RMSE from 49.6mm to 43.3mm versus the shared default, with no regression
# on any other tier since it's applied only here.
VIDEO_SOLVER_CONFIG = SolverConfig(
    stages=[
        StageSpec("global_only", ["global"], iters=80, w_data=1.0, w_prior=0.05,
                  w_limits=0.0, w_shape=0.0, torso_only=True),
        StageSpec("shape", ["global", "betas"], iters=80, w_data=1.0, w_prior=0.3,
                  w_limits=0.1, w_shape=1.0, torso_only=True),
        StageSpec("body_pose", ["global", "betas", "body_pose"], iters=200, w_data=3.0,
                  w_prior=0.05, w_limits=0.3, w_shape=0.5),
        StageSpec("temporal_polish", ["global", "betas", "body_pose"], iters=100, w_data=2.0,
                  w_prior=0.04, w_limits=0.3, w_shape=0.3, w_smooth=1.0, w_accel=0.3),
    ],
    sigma_data_m=0.1,
)


class UnsupportedFileError(ValueError):
    pass


def detect_file_kind(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".npz":
        return "npz"
    if ext == ".c3d":
        return "c3d"
    if ext == ".bvh":
        return "bvh"
    if ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    raise UnsupportedFileError(
        f"Unsupported file type {ext!r}. Supported: .npz, .c3d, .bvh, {', '.join(SUPPORTED_VIDEO_EXTS)}"
    )


@dataclass
class RawCapture:
    """Whatever the source format directly measured, before SMPL-X
    unification — this is what the "Raw Mocap" tab displays."""

    points: np.ndarray            # (T, K, 3)
    conf: np.ndarray              # (T, K)
    labels: List[str]
    edges: List[Tuple[int, int]]  # bone/marker connectivity, [] if unknown
    fps: float
    kind: str                     # "smplx_params" | "c3d_markers" | "bvh_skeleton" | "video_mediapipe"
    notes: List[str] = field(default_factory=list)


@dataclass
class FitResult:
    motion: Optional[CanonicalMotion]
    quality: Optional[FitQuality]
    notes: List[str] = field(default_factory=list)
    unavailable_reason: Optional[str] = None


def _fit_frame_selection(n: int, fps: float, target_count: Optional[int]) -> "tuple[np.ndarray, int, float]":
    """Indices for evenly subsampling an n-frame clip down to
    ~target_count frames, spread across the *whole* clip, for the
    expensive fitting/rendering step only — never for the raw-mocap view.

    This is deliberately not "the first target_count frames": that
    silently threw away everything after the first couple of seconds of a
    longer capture. A real file in this project's own test corpus runs
    378-620 frames at 100 fps (4-6 seconds); taking only the first 200
    (or, at the UI's old default, the first 30) frames meant the SMPL-X
    Fit tab could show only the opening fraction of a second of a longer
    clip's motion (reported bug). Evenly-spaced sampling instead keeps a
    representative view of the entire motion, and adjusts `fps` to match
    exactly (a real integer step, not an approximation), so playback
    timing on the subsampled fit stays correct.

    Returns (indices, step, effective_fps).
    """
    if target_count is None or n <= target_count:
        return np.arange(n), 1, fps
    step = max(1, n // target_count)
    idx = np.arange(0, n, step)[:target_count]
    return idx, step, fps / step


def subsample_for_fit(points: np.ndarray, conf: np.ndarray, fps: float,
                       target_count: Optional[int]) -> "tuple[np.ndarray, np.ndarray, float, int]":
    """Apply `_fit_frame_selection` to a (T, K, ...) points/conf pair.

    Returns (points, conf, fit_fps, step). Callers that then build a
    `SolverConfig` for `fit_observation` should pass `step` through
    `solver.scale_temporal_weights_for_subsampling` — see that function's
    docstring for the real bug this fixes (subsampled clips otherwise get
    their genuine motion penalized as implausible jitter).
    """
    idx, step, fit_fps = _fit_frame_selection(points.shape[0], fps, target_count)
    return points[idx], conf[idx], fit_fps, step


def process_npz(file_bytes: bytes, filename: str, body: SMPLXBody,
                 max_fit_frames: Optional[int] = 60) -> Tuple[RawCapture, FitResult]:
    try:
        with np.load(io.BytesIO(file_bytes), allow_pickle=True) as npz:
            data = dict(npz)
    except Exception as e:
        raise UnsupportedFileError(f"Could not read {filename!r} as a .npz file: {e}") from e

    try:
        motion = ingest_params(data, source_name=filename)
    except (UnsupportedFormatError, ValueError, KeyError) as e:
        raise UnsupportedFileError(
            f"{filename!r} is a valid .npz file but not a recognized SMPL/SMPL-H/SMPL-X "
            f"parameter format: {e}"
        ) from e

    import torch

    # Raw skeleton view: the COMPLETE clip, always — computing joint
    # positions via forward kinematics has no per-frame optimization cost,
    # so there is no reason to ever truncate what the user can see here
    # (see subsample_for_fit's docstring for the bug this fixes).
    t_full = motion.num_frames
    full_params = BodyParams(
        betas=torch.tensor(motion.betas), global_orient=torch.tensor(motion.global_orient),
        body_pose=torch.tensor(motion.body_pose), left_hand_pose=torch.tensor(motion.left_hand_pose),
        right_hand_pose=torch.tensor(motion.right_hand_pose), transl=torch.tensor(motion.trans),
    )
    with torch.no_grad():
        full_out = body.forward(full_params, return_verts=False)
    joints = full_out.joints.numpy()[:, :55, :]  # kinematic-tree joints only, for skeleton display

    from smplx.joint_names import JOINT_NAMES
    from app.render3d import skeleton_edges_from_parents

    notes = [f"Source format: {motion.meta.get('source_format', 'unknown')}"]
    if motion.part_mask[:, 2].max() == 0:
        notes.append("Hands unobserved in source (frozen rest pose, or absent) — shown at rest.")
    if motion.part_mask[:, 4].max() == 0:
        notes.append("Face unobserved (out of scope of the 159-d canonical layout).")

    raw = RawCapture(
        points=joints, conf=np.ones((t_full, 55)), labels=list(JOINT_NAMES[:55]),
        edges=skeleton_edges_from_parents(body.layer.parents.numpy()),
        fps=motion.fps, kind="smplx_params", notes=notes,
    )

    # SMPL-X Fit tab: subsampled only for the mesh-rendering step, which
    # does have a real per-frame cost (a full pyrender render each frame).
    idx, _, fit_fps = _fit_frame_selection(t_full, motion.fps, max_fit_frames)
    fit_motion = CanonicalMotion(
        trans=motion.trans[idx], global_orient=motion.global_orient[idx],
        body_pose=motion.body_pose[idx], left_hand_pose=motion.left_hand_pose[idx],
        right_hand_pose=motion.right_hand_pose[idx], betas=motion.betas,
        part_mask=motion.part_mask[idx], fps=fit_fps, meta=motion.meta,
    )
    subsample_note = (
        f"Fit/mesh tab shows {len(idx)} of {t_full} frames, evenly sampled across the whole clip "
        "(rendering every frame is unnecessary for a representative preview)."
        if len(idx) < t_full else None
    )
    quality = score_canonical_motion(fit_motion, joint_rmse_mm=float("nan"))
    fit_notes = notes + ["Tier 0: parameters converted directly (slice/reorder), no optimization needed."]
    if subsample_note:
        fit_notes.append(subsample_note)
    fit = FitResult(motion=fit_motion, quality=quality, notes=fit_notes)
    return raw, fit


def process_c3d(file_bytes: bytes, filename: str, body: SMPLXBody,
                 max_fit_frames: Optional[int] = 60) -> Tuple[RawCapture, FitResult]:
    try:
        cap = load_c3d(io.BytesIO(file_bytes))
    except Exception as e:
        raise UnsupportedFileError(f"Could not read {filename!r} as a .c3d file: {e}") from e

    # Raw marker view: the COMPLETE clip, always — see subsample_for_fit's
    # docstring for the bug this fixes (a fixed frame cap silently
    # truncated longer real captures to a fraction of a second).
    points = cap.points_m
    conf_full = np.ones(points.shape[:2])
    t_full = points.shape[0]
    raw = RawCapture(
        points=points, conf=conf_full, labels=cap.labels, edges=[],
        fps=cap.fps, kind="c3d_markers",
        notes=[f"{len(cap.labels)} raw markers, {t_full} frames" +
               (" (up-axis auto-detected)" if not cap.up_axis_guessed else " (up-axis guessed, low confidence)")],
    )

    try:
        layout, cols = build_matched_layout(cap.labels)
    except ValueError as e:
        return raw, FitResult(motion=None, quality=None, unavailable_reason=str(e))

    fit_points_full = points[:, cols, :]
    fit_conf_full = np.ones((t_full, len(cols)))
    fit_points, fit_conf, fit_fps, step = subsample_for_fit(fit_points_full, fit_conf_full, cap.fps, max_fit_frames)
    observation = Observation(points=fit_points, conf=fit_conf, joint_names=layout.point_names,
                               space="world3d", fps=fit_fps, up_axis=cap.up_axis)
    fit_cfg = SolverConfig(stages=scale_temporal_weights_for_subsampling(SolverConfig().stages, step))
    motion, quality = fit_observation(observation, body, layout, config=fit_cfg)
    notes = [f"Fit using {layout.num_points}/{len(cap.labels)} markers matched to the VICON_50 protocol "
             "(VertexCorr onto the real SMPL-X mesh surface)."]
    if fit_points.shape[0] < t_full:
        notes.append(f"Fit/mesh tab shows {fit_points.shape[0]} of {t_full} frames, evenly sampled "
                     "across the whole clip (rendering every frame is unnecessary for a representative preview).")
    return raw, FitResult(motion=motion, quality=quality, notes=notes)


def process_bvh(file_bytes: bytes, filename: str, body: SMPLXBody,
                 max_fit_frames: Optional[int] = 60) -> Tuple[RawCapture, FitResult]:
    try:
        cap = load_bvh(io.BytesIO(file_bytes))
    except Exception as e:
        raise UnsupportedFileError(f"Could not read {filename!r} as a .bvh file: {e}") from e

    # Raw skeleton view: the COMPLETE clip, always — see
    # subsample_for_fit's docstring for the bug this fixes.
    edges = [(i, p) for i, p in enumerate(cap.parents) if p >= 0]
    points = cap.points_m
    t_full = points.shape[0]
    axis_note = "up-axis auto-detected" if not cap.up_axis_guessed else "up-axis guessed, low confidence"
    units_note = "units auto-scaled to a plausible body height" if cap.units_guessed else "units taken as meters"
    raw = RawCapture(
        points=points, conf=np.ones((t_full, len(cap.joint_names))), labels=cap.joint_names, edges=edges,
        fps=cap.fps, kind="bvh_skeleton",
        notes=[f"{len(cap.joint_names)} BVH joints, {t_full} frames ({axis_note}; {units_note} — BVH files "
               "carry no standard unit, see src/smplx/adapters/bvh.py)."],
    )

    try:
        layout, cols = match_bvh_layout(cap.joint_names)
    except ValueError as e:
        return raw, FitResult(motion=None, quality=None, unavailable_reason=str(e))

    fit_points_full = points[:, cols, :]
    fit_conf_full = np.ones((t_full, len(cols)))
    fit_points, fit_conf, fit_fps, step = subsample_for_fit(fit_points_full, fit_conf_full, cap.fps, max_fit_frames)
    observation = Observation(points=fit_points, conf=fit_conf, joint_names=layout.point_names,
                               space="world3d", fps=fit_fps, up_axis=cap.up_axis)
    fit_cfg = SolverConfig(stages=scale_temporal_weights_for_subsampling(SolverConfig().stages, step))
    motion, quality = fit_observation(observation, body, layout, config=fit_cfg)
    notes = [f"Fit using {layout.num_points}/{len(cap.joint_names)} BVH joints matched by name "
             "(CMU/Mixamo-style naming convention, see adapters/bvh.py)."]
    if fit_points.shape[0] < t_full:
        notes.append(f"Fit/mesh tab shows {fit_points.shape[0]} of {t_full} frames, evenly sampled "
                     "across the whole clip (rendering every frame is unnecessary for a representative preview).")
    return raw, FitResult(motion=motion, quality=quality, notes=notes)


def process_video(file_bytes: bytes, filename: str, body: SMPLXBody,
                   max_extract_frames: Optional[int] = 150,
                   max_fit_frames: Optional[int] = 60,
                   progress_callback=None) -> Tuple[RawCapture, FitResult]:
    """`max_extract_frames` bounds MediaPipe extraction itself (the one
    genuinely per-frame-expensive step for video — there is no way to see
    a video frame's pose without running the model on it, unlike the
    other formats' raw views, which are always shown in full). The raw
    tab shows every frame that *was* extracted; `max_fit_frames` then
    independently subsamples across the whole extracted range for the
    fitting/mesh-rendering step, same as the other formats."""
    from app.video_extract import extract_pose_from_video, get_mediapipe_skeleton_edges

    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        cap = extract_pose_from_video(tmp_path, max_frames=max_extract_frames, progress_callback=progress_callback)
    except Exception as e:
        raise UnsupportedFileError(f"Could not process {filename!r} as a video: {e}") from e
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    t_full = cap.points_m.shape[0]
    observed_fraction = float((cap.conf > 0.3).mean())
    notes = [f"MediaPipe Pose world-landmark extraction, {t_full} frames @ {cap.fps:.1f} fps "
             f"(capped at {max_extract_frames} frames of the source video, since extraction runs a neural "
             "network per frame).",
             f"{observed_fraction * 100:.0f}% of landmarks confidently tracked (visibility > 0.3)."]
    raw = RawCapture(
        points=cap.points_m, conf=cap.conf, labels=list(MEDIAPIPE_33.point_names),
        edges=get_mediapipe_skeleton_edges(), fps=cap.fps, kind="video_mediapipe", notes=notes,
    )

    if observed_fraction < 0.1:
        return raw, FitResult(
            motion=None, quality=None,
            unavailable_reason="MediaPipe could not confidently track a person in this video "
                                "(landmark visibility too low); no SMPL-X fit was attempted.",
        )

    conf_full = np.where(cap.conf > 0.3, cap.conf, 0.0)  # low-confidence landmarks treated as unobserved
    fit_points, fit_conf, fit_fps, step = subsample_for_fit(cap.points_m, conf_full, cap.fps, max_fit_frames)
    observation = Observation(points=fit_points, conf=fit_conf, joint_names=MEDIAPIPE_33.point_names,
                               space="world3d", fps=fit_fps, up_axis="z")
    fit_cfg = SolverConfig(
        stages=scale_temporal_weights_for_subsampling(VIDEO_SOLVER_CONFIG.stages, step),
        sigma_data_m=VIDEO_SOLVER_CONFIG.sigma_data_m,
    )
    motion, quality = fit_observation(observation, body, MEDIAPIPE_33, config=fit_cfg)
    fit_notes = notes + ["Fit via the mediapipe_33 layout (surface-approximate joint correspondences, "
                          "reduced structural weight — see layouts/builtin.py)."]
    if fit_points.shape[0] < t_full:
        fit_notes.append(f"Fit/mesh tab shows {fit_points.shape[0]} of {t_full} extracted frames, evenly "
                          "sampled across the whole clip.")
    return raw, FitResult(motion=motion, quality=quality, notes=fit_notes)


def process_upload(file_bytes: bytes, filename: str, body: SMPLXBody,
                    max_fit_frames: Optional[int] = 60,
                    max_extract_frames: Optional[int] = 150,
                    progress_callback=None) -> Tuple[RawCapture, FitResult]:
    """`max_fit_frames` bounds the SMPL-X Fit tab's mesh-rendering/fitting
    step only (evenly subsampled across the whole clip) — the Raw Mocap
    tab always shows the complete clip for every format except video,
    where `max_extract_frames` bounds MediaPipe extraction itself (see
    process_video)."""
    kind = detect_file_kind(filename)
    if kind == "npz":
        return process_npz(file_bytes, filename, body, max_fit_frames=max_fit_frames)
    if kind == "c3d":
        return process_c3d(file_bytes, filename, body, max_fit_frames=max_fit_frames)
    if kind == "bvh":
        return process_bvh(file_bytes, filename, body, max_fit_frames=max_fit_frames)
    if kind == "video":
        return process_video(file_bytes, filename, body, max_extract_frames=max_extract_frames,
                              max_fit_frames=max_fit_frames, progress_callback=progress_callback)
    raise UnsupportedFileError(filename)  # pragma: no cover - detect_file_kind already validates
