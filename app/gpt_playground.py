"""Leonidas motion GPT playground — Streamlit app.

Upload a trained MoCapGPT checkpoint (a `best.pt`/`last.pt` from
`src/gpt/train.py`, by local file path — these are hundreds of MB, too
large for a comfortable browser upload) and a seed motion clip (`.npz`),
and watch the model continue the motion: the first N frames of the upload
seed the model's context, then it generates the next M frames on its own,
autoregressively. The whole sequence (real seed + generated continuation)
is rendered in the same interactive, mouse-orbitable 3D viewport as the
main mocap viewer app.

Run with:  streamlit run app/gpt_playground.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st
import torch

from app.model_bootstrap import ensure_smplx_model_downloaded
from app.render3d import build_mesh_player_threejs
from src.gpt.gpt import MoCapGPT
from src.gpt.train import _load_clip
from src.smplx.config import find_smplx_model
from src.smplx.features import model_features_to_canonical
from src.smplx.fitting.body import BodyParams, SMPLXBody
from src.smplx.schema import NUM_BETAS

st.set_page_config(page_title="Leonidas Motion GPT Playground", layout="wide")


@st.cache_resource(show_spinner="Loading SMPL-X body model...")
def get_body_model():
    return SMPLXBody()


def resolve_checkpoint_path(path_str: str) -> Path:
    """Accept either a direct file path or a directory containing one
    (same convenience as `src.smplx.config.find_smplx_model` for the body
    model) — "best" is preferred over "last" when both exist. Matches
    purely by stem, not extension, since a checkpoint downloaded from
    elsewhere (e.g. Kaggle) can end up renamed to `.zip`/`.txt`; torch's
    own `.pt` format is internally a zip archive regardless of what it's
    named, and `torch.load` doesn't care what extension the file has.

    Raises FileNotFoundError with a clear message if nothing is found.
    """
    p = Path(path_str)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"No file or directory found at {path_str!r}.")
    for stem in ("best", "last"):
        matches = sorted(p.glob(f"{stem}.*"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"{path_str!r} is a directory but no best.* or last.* checkpoint file was found inside it."
    )


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_checkpoint(path: str, _mtime: float, _size: int):
    """`_mtime`/`_size` (leading underscore -> excluded from the cache key's
    hashing, but still part of the key via Streamlit's default value-based
    caching since they're plain floats/ints) let a checkpoint overwritten
    at the same path invalidate the cache instead of silently reusing a
    stale loaded model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "model_config", "norm_mean", "norm_std"):
        if key not in ckpt:
            raise ValueError(
                f"{path!r} does not look like a checkpoint saved by src/gpt/train.py "
                f"(missing {key!r}) — point this at a best.pt or last.pt file."
            )
    model = MoCapGPT(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    # MoCapGPT auto-detects CUDA in __init__ and predict_next()/rollout()
    # always move their input to model.device — but construction alone
    # never moves the model's own weights there, so on a CUDA machine the
    # two silently end up on different devices. This viewer only ever
    # needs CPU (the models here are small; SMPL-X forward kinematics in
    # this app already runs on CPU too), so pin both explicitly.
    model.device = torch.device("cpu")
    model = model.to(model.device)
    model.eval()
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    return model, ckpt["model_config"], mean, std, ckpt.get("config")


@st.cache_data(show_spinner="Reading seed clip...")
def load_seed_clip(file_bytes: bytes, filename: str, target_fps: float):
    suffix = os.path.splitext(filename)[1] or ".npz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        out, reason = _load_clip(Path(tmp_path), target_fps, allowed_verdicts=[])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if out is None:
        raise ValueError(f"Could not read {filename!r} as a seed clip: {reason}")
    features, _part_mask = out
    return features  # (T, 315) float32, unnormalized model features


@torch.no_grad()
def generate(model: MoCapGPT, mean: torch.Tensor, std: torch.Tensor,
            seed_features: np.ndarray, n_generate: int) -> np.ndarray:
    """Seed with `seed_features` (T, 315) real/unnormalized model features,
    generate `n_generate` more frames autoregressively, and return the
    whole sequence (seed + generated), unnormalized, (T + n_generate, 315).
    """
    seed_t = torch.tensor(seed_features, dtype=torch.float32)  # matches the model's/mean's/std's own dtype
    seed_norm = (seed_t - mean) / std
    generated_norm = model.rollout(seed_norm, n_generate)  # (n_generate, 315), normalized
    generated = (generated_norm * std + mean).numpy()
    return np.concatenate([seed_features, generated], axis=0)


def render_generated_motion(body: SMPLXBody, combined_features: np.ndarray, n_seed: int,
                            fps: float, width: int, height: int):
    canonical = model_features_to_canonical(combined_features.astype(np.float64), initial_trans=np.zeros(3))
    t = canonical.shape[0]
    params = BodyParams(
        betas=torch.zeros(NUM_BETAS, dtype=torch.float64),
        global_orient=torch.tensor(canonical[:, 3:6]),
        body_pose=torch.tensor(canonical[:, 6:69]),
        left_hand_pose=torch.tensor(canonical[:, 69:114]),
        right_hand_pose=torch.tensor(canonical[:, 114:159]),
        transl=torch.tensor(canonical[:, 0:3]),
    )
    with torch.no_grad():
        out = body.forward(params, return_verts=True)
    vertices = out.vertices.numpy()
    faces = body.layer.faces.astype(np.int64)

    html = build_mesh_player_threejs(vertices, faces, fps=fps, width=width, height=height)
    st.components.v1.html(html, height=height + 110, scrolling=False)
    st.caption(f"Frames 0 to {n_seed - 1}: the real uploaded motion (the model's seed context). "
               f"Frames {n_seed} to {t - 1}: generated by the model on its own, one frame at a time. "
               "Body shape is shown neutral — the model does not predict it.")


def main():
    st.title("Leonidas Motion GPT Playground")
    st.caption("Seed a trained motion GPT with the start of a real clip and watch it generate the "
               "rest on its own, rendered as an SMPL-X character in an interactive 3D viewport.")

    ensure_smplx_model_downloaded()
    model_path = find_smplx_model()
    if model_path is None:
        st.error("No SMPL-X model file found. Set the LEONIDAS_SMPLX_MODEL environment variable "
                 "to a SMPLX_*.npz file (registration-gated, see https://smpl-x.is.tue.mpg.de).")
        st.stop()

    ckpt_path_str = st.sidebar.text_input(
        "GPT checkpoint path (best.pt/last.pt, or their run folder)", value="",
        help="A local file path, not a browser upload — checkpoints are typically hundreds of MB, "
             "produced by src/gpt/train.py (see src/gpt/evaluate_scaling.py for the same format). "
             "A directory (e.g. a run's output folder) is also accepted — best.* is used over last.* "
             "if both are present, matching by name only, not extension.",
    )
    uploaded = st.sidebar.file_uploader("Seed motion clip (.npz)", type=["npz"])
    n_context = st.sidebar.slider("Context frames (from the seed clip)", min_value=4, max_value=300,
                                  value=60, step=1,
                                  help="How many real frames from the start of the upload the model "
                                       "sees before it starts generating on its own.")
    n_generate = st.sidebar.slider("Frames to generate", min_value=1, max_value=300, value=60, step=1)

    if not ckpt_path_str:
        st.info("Enter a checkpoint path and upload a seed .npz clip from the sidebar to begin.")
        return
    try:
        ckpt_path = resolve_checkpoint_path(ckpt_path_str)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    try:
        stat = ckpt_path.stat()
        model, model_config, mean, std, train_cfg = load_checkpoint(str(ckpt_path), stat.st_mtime, stat.st_size)
    except Exception as e:
        st.error(f"Could not load checkpoint: {e}")
        return

    n_params = sum(p.numel() for p in model.parameters())
    cols = st.columns(4)
    cols[0].metric("Parameters", f"{n_params / 1e6:.2f}M")
    cols[1].metric("Layers", model_config["n_layers"])
    cols[2].metric("Embedding dim", model_config["n_embd"])
    cols[3].metric("Block size", model_config["block_size"])

    if uploaded is None:
        st.info("Upload a seed .npz clip from the sidebar to generate.")
        return

    target_fps = float(train_cfg["data"]["target_fps"]) if train_cfg else 30.0
    try:
        seed_features = load_seed_clip(uploaded.getvalue(), uploaded.name, target_fps)
    except ValueError as e:
        st.error(str(e))
        return

    if seed_features.shape[0] < n_context:
        st.warning(f"The uploaded clip only has {seed_features.shape[0]} frames at {target_fps:.0f} fps "
                   f"after resampling — using all of them as context instead of {n_context}.")
        n_context = seed_features.shape[0]
    seed_features = seed_features[:n_context]

    if n_context > model_config["block_size"]:
        st.caption(f"Note: this checkpoint's context window is {model_config['block_size']} frames — "
                   f"only the most recent {model_config['block_size']} of your {n_context} context "
                   "frames actually influence each generated step.")

    with st.expander("Render settings"):
        c1, c2 = st.columns(2)
        width = c1.select_slider("Viewport width", options=[480, 560, 640, 720, 860], value=640, key="gpt_width")
        height = c2.select_slider("Viewport height", options=[360, 420, 480, 540, 640], value=480, key="gpt_height")

    if st.button("Generate", type="primary"):
        with st.spinner(f"Generating {n_generate} frames..."):
            body = get_body_model()
            combined = generate(model, mean, std, seed_features, n_generate)
        render_generated_motion(body, combined, n_seed=n_context, fps=target_fps, width=width, height=height)


if __name__ == "__main__":
    main()
