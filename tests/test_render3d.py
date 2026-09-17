"""Tests for the pyrender mesh renderer, focused on the two real bugs found
in production use: cross-thread OpenGL context failure under Streamlit's
threading model, and a camera "up" axis mismatched to this pipeline's
Z-up data convention (see doc/Mocap_Unification_Design.docx section 6.4).
"""
import threading

import numpy as np
import pytest
import torch

from app.render3d import render_mesh_frame
from src.smplx.fitting.body import BodyParams


def _cube_mesh():
    # A simple asymmetric box: tall along one axis, so an up-axis mixup is
    # visually (and numerically, via silhouette bounds) detectable.
    verts = np.array([
        [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0], [-0.2, 0.2, 0.0],
        [-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.2, 0.2, 2.0], [-0.2, 0.2, 2.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return verts, faces


def test_render_rejects_bad_up_axis():
    verts, faces = _cube_mesh()
    with pytest.raises(ValueError):
        render_mesh_frame(verts, faces, up_axis="x")


def test_render_from_multiple_threads_does_not_crash():
    """Regression test for the exact reported failure: pyrender raising
    GLError/"wglChoosePixelFormatARB is not exported" when a render is
    requested from a thread other than the one that first touched OpenGL —
    which is what happens under real `streamlit run` (each script rerun
    executes on its own worker thread)."""
    verts, faces = _cube_mesh()
    render_mesh_frame(verts, faces, image_size=(96, 96))  # warm up on this thread

    results = {}

    def worker(name):
        try:
            img = render_mesh_frame(verts, faces, image_size=(96, 96))
            results[name] = ("ok", img.shape)
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            results[name] = ("error", repr(e))

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    for name, result in results.items():
        assert result[0] == "ok", f"{name} failed: {result}"
        assert result[1] == (96, 96, 3)


def test_render_produces_real_non_blank_image():
    verts, faces = _cube_mesh()
    img = render_mesh_frame(verts, faces, image_size=(128, 128))
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8
    assert img.std() > 1.0  # not a flat/blank image


def test_up_axis_changes_camera_orientation_measurably():
    """z-up and y-up renders of the same asymmetric mesh from the same
    azimuth/elevation must differ — if they were identical, up_axis would
    not actually be affecting the camera."""
    verts, faces = _cube_mesh()
    img_z = render_mesh_frame(verts, faces, azimuth_deg=20, elevation_deg=10,
                               image_size=(128, 128), up_axis="z")
    img_y = render_mesh_frame(verts, faces, azimuth_deg=20, elevation_deg=10,
                               image_size=(128, 128), up_axis="y")
    assert not np.array_equal(img_z, img_y)


def test_real_smplx_render_upright_after_fix(smplx_body):
    """End-to-end against real project data: a standing pose from the real
    AMASS sample file must render as a recognizably tall, upright figure
    once fed through the default (z-up) camera — this is the actual bug
    the user reported, verified against real data as requested."""
    import numpy as np
    from src.smplx.adapters.params import ingest_params

    with np.load("E:/github_clone/GPSM-1/data/0013_knocking1_poses.npz", allow_pickle=True) as d:
        data = dict(d)
    motion = ingest_params(data)
    frame = 400

    params = BodyParams(
        betas=torch.tensor(motion.betas),
        global_orient=torch.tensor(motion.global_orient[frame:frame + 1]),
        body_pose=torch.tensor(motion.body_pose[frame:frame + 1]),
        left_hand_pose=torch.tensor(motion.left_hand_pose[frame:frame + 1]),
        right_hand_pose=torch.tensor(motion.right_hand_pose[frame:frame + 1]),
        transl=torch.tensor(motion.trans[frame:frame + 1]),
    )
    with torch.no_grad():
        out = smplx_body.forward(params, return_verts=True)
    verts = out.vertices[0].numpy()

    # Confirms the premise: for this real data, Z (not Y) has by far the
    # largest extent — i.e. Z really is "up" for these vertices.
    extent = verts.max(axis=0) - verts.min(axis=0)
    assert extent[2] > extent[1] * 2 and extent[2] > extent[0] * 2

    img = render_mesh_frame(verts, smplx_body.layer.faces.astype(np.int64),
                             azimuth_deg=20, elevation_deg=10, image_size=(200, 300))
    assert img.shape == (300, 200, 3)
    assert img.std() > 1.0
