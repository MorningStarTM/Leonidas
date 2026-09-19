import os

import numpy as np
import pytest

from app.pipeline import UnsupportedFileError, detect_file_kind, process_upload

REAL_NPZ = "E:/github_clone/GPSM-1/data/0013_knocking1_poses.npz"
REAL_C3D = "E:/github_clone/GPSM-1/data/01_01.c3d"
REAL_BVH = "C:/Users/Ernest/Downloads/MoCap/Male1_bvh/Male1_A10_LieToCrouch.bvh"
NO_PERSON_VIDEO = "E:/github_clone/motion-matching-env/tools/debug_plots/mocap_simulation.mp4"


def test_detect_file_kind():
    assert detect_file_kind("clip.npz") == "npz"
    assert detect_file_kind("clip.C3D") == "c3d"
    assert detect_file_kind("clip.BVH") == "bvh"
    assert detect_file_kind("clip.mp4") == "video"
    assert detect_file_kind("clip.MOV") == "video"
    with pytest.raises(UnsupportedFileError):
        detect_file_kind("clip.txt")


@pytest.mark.skipif(not os.path.isfile(REAL_NPZ), reason="real npz sample not found")
def test_process_upload_npz(smplx_body):
    """Regression test for the reported bug: the Raw Mocap tab must show
    the *complete* clip regardless of the fit-tab frame budget — this
    real file has 853 frames; passing max_fit_frames=10 must not truncate
    the raw view down to 10 (the old, buggy behavior)."""
    with open(REAL_NPZ, "rb") as f:
        data = f.read()
    raw, fit = process_upload(data, os.path.basename(REAL_NPZ), smplx_body, max_fit_frames=10)

    assert raw.kind == "smplx_params"
    assert raw.points.shape[0] == 853, "raw tab must show the full clip, not the fit-frame budget"
    assert len(raw.edges) > 0
    assert not np.isnan(raw.points).any()

    assert fit.motion is not None
    assert fit.motion.num_frames == 10, "fit/mesh tab should respect the frame budget"
    assert not np.isnan(fit.motion.to_array()).any()
    assert fit.quality.verdict in ("GOOD", "DEGRADED", "REJECT")


def test_fit_frame_selection_spreads_across_whole_clip_not_just_the_start():
    """Precise, direct test of the subsampling primitive: the indices it
    picks must reach the *end* of a long clip, not cluster at the start —
    that's the exact property that fixes the reported bug (a longer
    capture's later motion never showing up in the SMPL-X Fit tab even
    though a frame budget was nominally being respected)."""
    from app.pipeline import _fit_frame_selection

    idx, step, fit_fps = _fit_frame_selection(n=853, fps=100.0, target_count=10)
    assert len(idx) == 10
    assert idx[0] == 0
    # The last picked index must be well into the back half of the clip —
    # a bug that took "the first 10 of 853" would leave it at 9.
    assert idx[-1] > 853 * 0.8, f"subsampling did not reach near the end of the clip: {idx}"
    assert step == 853 // 10
    assert fit_fps == pytest.approx(100.0 / step)


def test_fit_frame_selection_is_a_noop_when_clip_is_shorter_than_budget():
    from app.pipeline import _fit_frame_selection

    idx, step, fit_fps = _fit_frame_selection(n=8, fps=30.0, target_count=60)
    assert list(idx) == list(range(8))
    assert step == 1
    assert fit_fps == 30.0


def test_fit_frame_selection_handles_none_budget_as_no_limit():
    from app.pipeline import _fit_frame_selection

    idx, step, fit_fps = _fit_frame_selection(n=500, fps=60.0, target_count=None)
    assert len(idx) == 500
    assert fit_fps == 60.0


def test_subsample_for_fit_returns_step_for_temporal_weight_scaling():
    """`step` must be exposed so callers can pass it to
    `solver.scale_temporal_weights_for_subsampling` — without it, a
    subsampled clip's genuine motion gets penalized by the smoothness
    loss as if it were implausible jitter (see that function's docstring
    for the real bug this fixes)."""
    from app.pipeline import subsample_for_fit

    points = np.zeros((100, 5, 3))
    conf = np.ones((100, 5))
    out_points, out_conf, fit_fps, step = subsample_for_fit(points, conf, fps=100.0, target_count=20)
    assert out_points.shape[0] == 20
    assert step == 5
    assert fit_fps == 20.0


@pytest.mark.skipif(not os.path.isfile(REAL_NPZ), reason="real npz sample not found")
def test_process_upload_rejects_garbage_npz(smplx_body):
    import io
    bad = io.BytesIO()
    np.savez(bad, something_unrelated=np.zeros(5))
    with pytest.raises(UnsupportedFileError):
        process_upload(bad.getvalue(), "bad.npz", smplx_body)


@pytest.mark.skipif(not os.path.isfile(REAL_C3D), reason="real c3d sample not found")
def test_process_upload_c3d(smplx_body):
    """This real file has 2751 frames — the raw tab must show all of them
    regardless of the (much smaller) fit-tab frame budget."""
    with open(REAL_C3D, "rb") as f:
        data = f.read()
    raw, fit = process_upload(data, os.path.basename(REAL_C3D), smplx_body, max_fit_frames=12)

    assert raw.kind == "c3d_markers"
    assert raw.points.shape == (2751, len(raw.labels), 3), "raw tab must show the full clip"
    assert not np.isnan(raw.points).any()

    assert fit.motion is not None  # this file has enough VICON_50 overlap (see test_markers_adapter)
    assert fit.motion.num_frames == 12
    assert not np.isnan(fit.motion.to_array()).any()


@pytest.mark.skipif(not os.path.isfile(REAL_BVH), reason="real bvh sample not found")
def test_process_upload_bvh(smplx_body):
    """This real file has 196 frames — the raw tab must show all of them
    regardless of the (much smaller) fit-tab frame budget."""
    with open(REAL_BVH, "rb") as f:
        data = f.read()
    raw, fit = process_upload(data, os.path.basename(REAL_BVH), smplx_body, max_fit_frames=10)

    assert raw.kind == "bvh_skeleton"
    assert raw.points.shape[0] == 196, "raw tab must show the full clip"
    assert len(raw.edges) > 0
    assert not np.isnan(raw.points).any()

    assert fit.motion is not None
    assert fit.motion.num_frames == 10
    assert not np.isnan(fit.motion.to_array()).any()
    assert fit.quality.verdict in ("GOOD", "DEGRADED", "REJECT")


@pytest.mark.skipif(not os.path.isfile(NO_PERSON_VIDEO), reason="sample video not found")
def test_process_upload_video_graceful_no_person(smplx_body):
    """A video with no detectable human must not crash the app — it should
    surface a clear reason instead. Video is the one format where the raw
    tab's frame count is legitimately capped, since extraction itself
    (a neural-network pass per frame) is the expensive step."""
    with open(NO_PERSON_VIDEO, "rb") as f:
        data = f.read()
    raw, fit = process_upload(data, "clip.mp4", smplx_body, max_extract_frames=8)
    assert raw.kind == "video_mediapipe"
    assert raw.points.shape[0] == 8
    assert fit.motion is None
    assert fit.unavailable_reason is not None
