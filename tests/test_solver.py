import numpy as np
import torch

from src.smplx.fitting.body import BodyParams
from src.smplx.fitting.solver import fit_observation
from src.smplx.layouts import get_layout
from src.smplx.schema import Observation


def _synthetic_clip(body, layout, rng, t=10):
    base_pose = rng.normal(0, 0.25, 63)
    body_pose_gt = np.stack([base_pose + rng.normal(0, 0.03, 63) for _ in range(t)])
    global_orient_gt = np.stack([rng.normal(0, 0.15, 3) for _ in range(t)])
    transl_gt = np.cumsum(rng.normal(0, 0.02, (t, 3)), axis=0)
    betas_gt = rng.normal(0, 0.5, 10)

    params_gt = BodyParams(
        betas=torch.tensor(betas_gt),
        global_orient=torch.tensor(global_orient_gt),
        body_pose=torch.tensor(body_pose_gt),
        left_hand_pose=torch.zeros(t, 45, dtype=torch.float64),
        right_hand_pose=torch.zeros(t, 45, dtype=torch.float64),
        transl=torch.tensor(transl_gt),
    )
    with torch.no_grad():
        joints_gt = body.forward(params_gt).joints.numpy()

    corr_idx = layout.smplx_indices()
    return {
        "joints_gt": joints_gt,
        "corr_idx": corr_idx,
        "body_pose_gt": body_pose_gt,
        "global_orient_gt": global_orient_gt,
        "transl_gt": transl_gt,
    }


def test_fit_observation_recovers_synthetic_pose(smplx_body):
    rng = np.random.default_rng(42)
    layout = get_layout("h36m_17")
    t = 12
    clip = _synthetic_clip(smplx_body, layout, rng, t=t)

    target_clean = clip["joints_gt"][:, clip["corr_idx"], :]
    points = target_clean + rng.normal(0, 0.01, target_clean.shape)
    conf = np.ones((t, len(clip["corr_idx"])))

    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=30.0, up_axis="z")

    motion, quality = fit_observation(obs, smplx_body, layout)

    assert np.abs(motion.global_orient - clip["global_orient_gt"]).mean() < 0.15
    assert np.abs(motion.body_pose - clip["body_pose_gt"]).mean() < 0.15
    assert np.abs(motion.trans - clip["transl_gt"]).mean() < 0.1
    assert quality.verdict in ("GOOD", "DEGRADED")
    assert not np.isnan(motion.to_array()).any()


def test_fit_observation_handles_missing_limb_via_prior(smplx_body):
    """Directly exercises the "zero/absent should never be silently
    encoded, and must still yield a complete, plausible pose" requirement:
    an entire joint is unobserved for part of the clip; the solver must
    fill it with a plausible value (via the prior) rather than diverging,
    and must mark it unobserved in part_mask/quality rather than pretend
    it was measured."""
    rng = np.random.default_rng(7)
    layout = get_layout("h36m_17")
    t = 10
    clip = _synthetic_clip(smplx_body, layout, rng, t=t)
    target_clean = clip["joints_gt"][:, clip["corr_idx"], :]

    points = target_clean + rng.normal(0, 0.01, target_clean.shape)
    conf = np.ones((t, len(clip["corr_idx"])))

    wrist_col = layout.point_names.index("RWrist")
    conf[:, wrist_col] = 0.0
    points[:, wrist_col] = 0.0  # a naive pipeline would treat this as "at the origin"

    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=30.0, up_axis="z")
    motion, quality = fit_observation(obs, smplx_body, layout)

    # The overall fit should still succeed and stay physically plausible —
    # not collapse toward the zeroed-out wrist observation.
    assert np.abs(motion.body_pose - clip["body_pose_gt"]).mean() < 0.2
    assert quality.observed_fraction < 1.0  # correctly reflects the gap
    assert not np.isnan(motion.to_array()).any()


def test_hands_always_marked_unobserved_by_tier1_solver(smplx_body):
    """The built-in layouts have no per-finger correspondences, so hand
    pose is never actually fit — it must stay at rest and be marked
    unobserved, never presented as a real result (see solver.py docstring)."""
    rng = np.random.default_rng(1)
    layout = get_layout("coco_17")
    t = 6
    clip = _synthetic_clip(smplx_body, layout, rng, t=t)
    target_clean = clip["joints_gt"][:, clip["corr_idx"], :]
    obs = Observation(points=target_clean, conf=np.ones((t, len(clip["corr_idx"]))),
                       joint_names=layout.point_names, space="world3d", fps=30.0, up_axis="z")
    motion, _ = fit_observation(obs, smplx_body, layout)
    assert np.all(motion.left_hand_pose == 0.0)
    assert np.all(motion.right_hand_pose == 0.0)
    assert motion.part_mask[:, 2].max() == 0.0
    assert motion.part_mask[:, 3].max() == 0.0


def test_fit_observation_with_vertex_layout_vicon_50(smplx_body):
    """End-to-end recovery using the real VICON_50 marker->vertex
    correspondence table (VertexCorr), not just JointCorr layouts."""
    rng = np.random.default_rng(9)
    layout = get_layout("vicon_50")
    t = 8

    base_pose = rng.normal(0, 0.2, 63)
    body_pose_gt = np.stack([base_pose + rng.normal(0, 0.02, 63) for _ in range(t)])
    global_orient_gt = np.stack([rng.normal(0, 0.1, 3) for _ in range(t)])
    transl_gt = np.cumsum(rng.normal(0, 0.01, (t, 3)), axis=0)
    betas_gt = rng.normal(0, 0.4, 10)

    params_gt = BodyParams(
        betas=torch.tensor(betas_gt),
        global_orient=torch.tensor(global_orient_gt),
        body_pose=torch.tensor(body_pose_gt),
        left_hand_pose=torch.zeros(t, 45, dtype=torch.float64),
        right_hand_pose=torch.zeros(t, 45, dtype=torch.float64),
        transl=torch.tensor(transl_gt),
    )
    with torch.no_grad():
        out_gt = smplx_body.forward(params_gt, return_verts=True)
        vertex_ids = layout.target_indices()
        target_clean = out_gt.vertices[:, vertex_ids, :].numpy()

    points = target_clean + rng.normal(0, 0.005, target_clean.shape)
    conf = np.ones((t, len(vertex_ids)))
    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=30.0, up_axis="z")

    motion, quality = fit_observation(obs, smplx_body, layout)

    assert np.abs(motion.body_pose - body_pose_gt).mean() < 0.2
    assert np.abs(motion.global_orient - global_orient_gt).mean() < 0.2
    assert not np.isnan(motion.to_array()).any()
    assert quality.observed_fraction > 0.9


def test_rejects_2d_observation():
    import pytest
    layout = get_layout("coco_17")
    obs = Observation(
        points=np.zeros((5, 17, 2)), conf=np.ones((5, 17)),
        joint_names=layout.point_names, space="image2d", fps=30.0,
    )
    with pytest.raises(ValueError):
        fit_observation(obs, None, layout)
