"""Config-driven GPT training on unified SMPL-X motion. One run per config.

Reads clips from a folder (recursively) in either format:
  * MotionPrep output: .npz with trans, global_orient, body_pose,
    left_hand_pose, right_hand_pose (+ optional part_mask, fps, verdict)
  * raw AMASS / SMPL-H / SMPL-X parameter .npz (converted by ingest_params)
and trains MoCapGPT to predict the next frame at every position of a window.

Local:
    python -m src.gpt.train --config configs/gpt_sample.yaml
    python -m src.gpt.train --config configs/gpt_sample.yaml --set data.dir=/path model.n_layers=8
    python -m src.gpt.train --config configs/gpt_sample.yaml --count_params

Kaggle notebook (one notebook per model size):
    !pip install -q loguru pyyaml safetensors
    !git clone https://github.com/MorningStarTM/Leonidas.git
    %cd Leonidas
    !python -m src.gpt.train --config configs/gpt_sample.yaml \
        --set run.name=gpt_s data.dir=/kaggle/input/YOUR_DATASET run.output_dir=/kaggle/working/runs

Outputs in <output_dir>/<name>/: config.yaml, train_log.jsonl, best.pt,
last.pt (resume with run.resume_from=<path to last.pt>), results.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.gpt.gpt import MoCapGPT
from src.smplx.features import MODEL_FEATURE_DIM, canonical_to_model_features
from src.smplx.ops import resample_pose_sequence
from src.smplx.schema import NUM_PARTS

DEFAULTS: Dict[str, Any] = {
    "run": {
        "name": "gpt_run",
        "output_dir": "runs",
        "seed": 42,
        "resume_from": None,
        "max_train_minutes": 0,
        "device": "auto",
    },
    "data": {
        "dir": None,
        "target_fps": 30.0,
        "val_fraction": 0.05,
        "allowed_verdicts": ["GOOD", "DEGRADED"],
        "use_part_mask": True,
        "max_clips": None,
    },
    "model": {
        "n_embd": 256,
        "n_head": 8,
        "n_layers": 6,
        "dropout": 0.1,
        "block_size": 128,
    },
    "train": {
        "batch_size": 64,
        "grad_accum_steps": 1,
        "max_steps": 10000,
        "learning_rate": 3.0e-4,
        "min_lr_ratio": 0.1,
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "adam_betas": [0.9, 0.95],
        "grad_clip_norm": 1.0,
        "amp": "fp16",
        "eval_every": 500,
        "eval_batches": 50,
        "save_every": 1000,
        "log_every": 50,
        "patience_evals": 0,
    },
}

_CANON_KEYS = ("trans", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose")

# part_mask columns (trans, body, left_hand, right_hand, face) -> feature dims
# in the 315-d layout: [0:3] root velocity, [3:135] global_orient + body,
# [135:225] left hand, [225:315] right hand.
_PART_FEATURE_SLICES = [(0, 3), (3, 135), (135, 225), (225, 315), (0, 0)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _coerce_number(default: Any, value: Any, name: str) -> Any:
    # PyYAML reads `1e-4` (no decimal point) as a string, so numbers are
    # converted here instead of failing later inside the optimizer.
    if isinstance(default, bool) or not isinstance(default, (int, float)) or not isinstance(value, str):
        return value
    try:
        num = float(value)
    except ValueError:
        raise ValueError(f"Config value '{name}' must be a number, got {value!r}") from None
    return int(num) if isinstance(default, int) and num.is_integer() else num


def _deep_merge(base: Dict[str, Any], new: Optional[Dict[str, Any]], path: str = "") -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if key not in base:
            raise KeyError(f"Unknown config key '{path}{key}'. Valid keys here: {sorted(base)}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{path}{key}' must be a mapping.")
            out[key] = _deep_merge(base[key], value, f"{path}{key}.")
        elif isinstance(base[key], list) and base[key] and isinstance(base[key][0], float) and isinstance(value, list):
            out[key] = [_coerce_number(base[key][0], v, f"{path}{key}") for v in value]
        else:
            out[key] = _coerce_number(base[key], value, f"{path}{key}")
    return out


def _parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like section.key=value")
        dotted, raw = pair.split("=", 1)
        parts = dotted.split(".")
        node = nested
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return nested


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Defaults <- YAML file (or `config` dict) <- `section.key=value` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    if config:
        cfg = _deep_merge(cfg, config)
    if overrides:
        cfg = _deep_merge(cfg, _parse_overrides(overrides))
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    if m["n_embd"] % m["n_head"] != 0:
        raise ValueError(f"model.n_embd ({m['n_embd']}) must be divisible by model.n_head ({m['n_head']}).")
    if m["block_size"] < 2:
        raise ValueError("model.block_size must be >= 2.")
    if t["amp"] not in ("fp16", "bf16", "none"):
        raise ValueError("train.amp must be one of: fp16, bf16, none.")
    if t["grad_accum_steps"] < 1 or t["batch_size"] < 1 or t["max_steps"] < 1:
        raise ValueError("train.batch_size, train.grad_accum_steps and train.max_steps must be >= 1.")
    if not 0.0 < d["val_fraction"] < 1.0:
        raise ValueError("data.val_fraction must be between 0 and 1.")


def _model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg["model"]
    return {
        "state_dim": MODEL_FEATURE_DIM,
        "n_embd": m["n_embd"],
        "n_head": m["n_head"],
        "n_layers": m["n_layers"],
        "block_size": m["block_size"],
        "dropout": m["dropout"],
        "max_timestep": m["block_size"],
        "learning_rate": cfg["train"]["learning_rate"],
    }


def count_params(cfg: Dict[str, Any]) -> int:
    model = MoCapGPT(_model_config(cfg))
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_clip(path: Path, target_fps: float, allowed_verdicts) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
    """Returns ((features (T,315) float32, part_mask (T,5) float32), "") or (None, reason)."""
    try:
        with np.load(path, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable ({type(e).__name__})"

    if allowed_verdicts and "verdict" in d and str(d["verdict"]) not in allowed_verdicts:
        return None, "verdict filtered"

    try:
        if all(k in d for k in _CANON_KEYS):
            arr = np.concatenate([np.asarray(d[k], dtype=np.float64) for k in _CANON_KEYS], axis=1)
            mask = np.asarray(d["part_mask"], dtype=np.float32) if "part_mask" in d else np.ones((arr.shape[0], NUM_PARTS), np.float32)
            fps = float(d["fps"]) if "fps" in d else target_fps
        elif "poses" in d:
            from src.smplx.adapters.params import ingest_params

            motion = ingest_params(d)
            arr, mask, fps = motion.to_array(), motion.part_mask.astype(np.float32), float(motion.fps)
        else:
            return None, "unrecognized keys"

        if arr.shape[1] != 159 or mask.shape != (arr.shape[0], NUM_PARTS):
            return None, "unexpected shape"
        if not np.isfinite(arr).all():
            return None, "non-finite values"

        if abs(fps - target_fps) > 1e-3:
            n_src = arr.shape[0]
            arr = resample_pose_sequence(arr, fps, target_fps)
            idx = np.minimum(np.round(np.arange(arr.shape[0]) * fps / target_fps).astype(int), n_src - 1)
            mask = mask[idx]

        feats = canonical_to_model_features(arr).astype(np.float32)
        return (feats, mask.astype(np.float32)), ""
    except Exception as e:  # noqa: BLE001
        return None, f"conversion failed ({type(e).__name__}: {e})"


class MotionCorpus:
    """All clips concatenated into one (N, 315) tensor; windows are index ranges."""

    def __init__(self, feats: List[np.ndarray], masks: List[np.ndarray], window: int):
        self.window = window
        lengths = np.array([f.shape[0] for f in feats], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if len(lengths) else np.zeros(0, np.int64)
        self.num_frames = int(lengths.sum())
        self.data = torch.from_numpy(np.concatenate(feats, axis=0)) if len(feats) else torch.zeros(0, MODEL_FEATURE_DIM)
        self.mask = torch.from_numpy(np.concatenate(masks, axis=0)) if len(masks) else torch.zeros(0, NUM_PARTS)
        # window needs window+1 frames (inputs plus the shifted targets)
        counts = np.maximum(lengths - window, 0)
        self._cum_end = np.cumsum(counts)
        self._cum_start = self._cum_end - counts
        self.num_windows = int(counts.sum())
        # non-overlapping windows that tile each clip, for a fixed validation set
        starts = []
        for off, n in zip(self.offsets, lengths):
            starts.extend(int(off) + j * window for j in range(int((n - 1) // window)))
        self.fixed_starts = np.array(starts, dtype=np.int64)

    def sample_starts(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        r = rng.integers(0, self.num_windows, size=batch_size)
        clip = np.searchsorted(self._cum_end, r, side="right")
        return self.offsets[clip] + (r - self._cum_start[clip])

    def gather(self, starts: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.from_numpy(starts[:, None] + np.arange(self.window + 1)[None, :])
        return self.data[idx], self.mask[idx]


def build_corpora(cfg: Dict[str, Any]):
    dcfg = cfg["data"]
    if not dcfg["dir"]:
        raise ValueError("data.dir is not set. Point it at the folder holding your .npz clips.")
    root = Path(dcfg["dir"])
    if not root.is_dir():
        raise FileNotFoundError(f"data.dir does not exist: {root}")
    files = sorted(root.rglob("*.npz"))
    if dcfg["max_clips"]:
        files = files[: int(dcfg["max_clips"])]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {root}")

    window = cfg["model"]["block_size"]
    clips, skipped = [], {}
    for i, f in enumerate(files, 1):
        out, reason = _load_clip(f, dcfg["target_fps"], dcfg["allowed_verdicts"])
        if out is None:
            skipped[reason] = skipped.get(reason, 0) + 1
        elif out[0].shape[0] < window + 1:
            skipped["shorter than block_size+1"] = skipped.get("shorter than block_size+1", 0) + 1
        else:
            clips.append(out)
        if i % 500 == 0:
            print(f"  loaded {i}/{len(files)} files")
    if len(clips) < 2:
        raise ValueError(f"Only {len(clips)} usable clip(s) out of {len(files)} (skipped: {skipped}). Need at least 2.")

    order = np.random.default_rng(cfg["run"]["seed"]).permutation(len(clips))
    n_val = min(max(1, int(round(len(clips) * dcfg["val_fraction"]))), len(clips) - 1)
    val_ids, train_ids = order[:n_val], order[n_val:]
    train = MotionCorpus([clips[i][0] for i in train_ids], [clips[i][1] for i in train_ids], window)
    val = MotionCorpus([clips[i][0] for i in val_ids], [clips[i][1] for i in val_ids], window)
    if len(val.fixed_starts) == 0:
        raise ValueError("The validation clips have no full window; lower model.block_size or raise data.val_fraction.")
    info = {"files_found": len(files), "clips_used": len(clips), "skipped": skipped,
            "train_clips": len(train_ids), "val_clips": len(val_ids),
            "train_frames": train.num_frames, "val_frames": val.num_frames,
            "train_hours": train.num_frames / dcfg["target_fps"] / 3600.0,
            "val_windows": int(len(val.fixed_starts))}
    return train, val, info


def _normalization(corpus: MotionCorpus) -> Tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(MODEL_FEATURE_DIM, dtype=torch.float64)
    total_sq = torch.zeros(MODEL_FEATURE_DIM, dtype=torch.float64)
    for chunk in torch.split(corpus.data, 1_000_000):
        c = chunk.double()
        total += c.sum(0)
        total_sq += (c * c).sum(0)
    n = max(corpus.num_frames, 1)
    mean = total / n
    var = (total_sq / n - mean * mean).clamp_min(0.0)
    std = var.sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


def _part_to_feature_map(use_part_mask: bool, device) -> Optional[torch.Tensor]:
    if not use_part_mask:
        return None
    w = torch.zeros(NUM_PARTS, MODEL_FEATURE_DIM)
    for p, (a, b) in enumerate(_PART_FEATURE_SLICES):
        w[p, a:b] = 1.0
    return w.to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _prepare(corpus: MotionCorpus, starts, mean, std, part_map, device):
    frames, mask = corpus.gather(starts)
    frames = (frames.to(device) - mean) / std
    x, y = frames[:, :-1], frames[:, 1:]
    w = mask[:, 1:].to(device) @ part_map if part_map is not None else torch.ones_like(y)
    return x, y, w


def _masked_mse_sums(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (((pred - y) ** 2) * w).sum(), w.sum()


@torch.no_grad()
def evaluate(model, corpus, mean, std, part_map, device, autocast_kwargs, batch_size, max_batches, mode="model") -> float:
    """Weighted mean squared error (normalized units) over fixed validation windows.
    mode: "model", "zero" (predict the training mean) or "copy" (repeat the current frame)."""
    starts = corpus.fixed_starts
    limit = len(starts) if max_batches <= 0 else min(len(starts), max_batches * batch_size)
    if limit < len(starts):
        starts = starts[np.linspace(0, len(starts) - 1, limit).astype(int)]
    if mode == "model":
        model.eval()
    se_total, w_total = 0.0, 0.0
    for i in range(0, len(starts), batch_size):
        x, y, w = _prepare(corpus, starts[i:i + batch_size], mean, std, part_map, device)
        if mode == "model":
            with torch.autocast(**autocast_kwargs):
                pred = model(x)
            pred = pred.float()
        elif mode == "zero":
            pred = torch.zeros_like(y)
        else:
            pred = x
        se, ws = _masked_mse_sums(pred, y, w)
        se_total += float(se)
        w_total += float(ws)
    return se_total / max(w_total, 1.0)


def _lr_at(step: int, t: Dict[str, Any]) -> float:
    warm = t["warmup_steps"]
    if step < warm:
        return t["learning_rate"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, t["max_steps"] - warm))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return t["learning_rate"] * (t["min_lr_ratio"] + (1.0 - t["min_lr_ratio"]) * cos)


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    r, t = cfg["run"], cfg["train"]
    torch.manual_seed(r["seed"])
    np.random.seed(r["seed"])

    device = torch.device("cuda" if (r["device"] == "auto" and torch.cuda.is_available()) else
                          ("cpu" if r["device"] == "auto" else r["device"]))
    amp_mode = t["amp"] if device.type == "cuda" else "none"
    if amp_mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("train.amp=bf16 is not supported on this GPU (Kaggle T4/P100 need fp16).")
    idle_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16  # unused when amp is off
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode, idle_dtype)
    autocast_kwargs = {"device_type": device.type, "dtype": amp_dtype, "enabled": amp_mode != "none"}

    out_dir = Path(r["output_dir"]) / r["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[data] loading clips from {cfg['data']['dir']}")
    train_c, val_c, info = build_corpora(cfg)
    mean, std = _normalization(train_c)
    mean, std = mean.to(device), std.to(device)
    part_map = _part_to_feature_map(cfg["data"]["use_part_mask"], device)
    print(f"[data] {info}")

    model = MoCapGPT(_model_config(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t["learning_rate"], betas=tuple(t["adam_betas"]),
    )
    scaler = _make_scaler(amp_mode == "fp16")

    frames_per_step = t["batch_size"] * t["grad_accum_steps"] * cfg["model"]["block_size"]
    total_frames = frames_per_step * t["max_steps"]
    print(f"[model] {n_params / 1e6:.2f}M parameters | {frames_per_step} frames/step | "
          f"{total_frames / max(train_c.num_frames, 1):.1f} passes over the training frames in total | "
          f"device={device} amp={amp_mode}")

    step, best_val, best_step, frames_seen = 0, math.inf, 0, 0
    if r["resume_from"]:
        ckpt = torch.load(r["resume_from"], map_location=device, weights_only=False)
        arch = lambda c: {k: v for k, v in c.items() if k != "learning_rate"}  # noqa: E731
        if arch(ckpt["model_config"]) != arch(_model_config(cfg)):
            raise ValueError("resume_from checkpoint was trained with a different model architecture.")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        step, best_val, best_step, frames_seen = ckpt["step"], ckpt["best_val"], ckpt["best_step"], ckpt["frames_seen"]
        print(f"[resume] continuing from step {step} (best val so far {best_val:.5f})")

    eb, bs = t["eval_batches"], t["batch_size"]
    baselines = {
        "baseline_val_loss_mean": evaluate(None, val_c, mean, std, part_map, device, autocast_kwargs, bs, eb, "zero"),
        "baseline_val_loss_copy_last": evaluate(None, val_c, mean, std, part_map, device, autocast_kwargs, bs, eb, "copy"),
    }
    print(f"[baselines] predict-the-mean={baselines['baseline_val_loss_mean']:.5f} "
          f"repeat-last-frame={baselines['baseline_val_loss_copy_last']:.5f}")

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")
    rng = np.random.default_rng([r["seed"], step])
    start_time = time.time()
    last_train_loss, last_val, bad_evals, stop_reason = float("nan"), float("nan"), 0, "max_steps"

    def checkpoint(path: Path):
        _save_atomic({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step, "best_val": best_val, "best_step": best_step, "frames_seen": frames_seen,
            "model_config": _model_config(cfg), "config": cfg,
            "norm_mean": mean.cpu(), "norm_std": std.cpu(),
        }, path)

    def write_results(finished: bool):
        res = {
            "name": r["name"], "finished": finished, "stopped_reason": stop_reason,
            "n_params": n_params, "n_params_millions": round(n_params / 1e6, 3),
            "steps_done": step, "max_steps": t["max_steps"], "frames_seen": frames_seen,
            "best_val_loss": None if math.isinf(best_val) else best_val, "best_step": best_step,
            "final_val_loss": last_val, "final_train_loss": last_train_loss,
            **baselines, **info,
            "wall_minutes_this_session": round((time.time() - start_time) / 60.0, 2),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu",
            "torch": torch.__version__, "config": cfg,
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        return res

    last_eval_step = -1

    def do_eval() -> None:
        nonlocal best_val, best_step, last_val, bad_evals, last_eval_step
        last_eval_step = step
        last_val = evaluate(model, val_c, mean, std, part_map, device, autocast_kwargs, bs, eb)
        rec = {"event": "eval", "step": step, "val_loss": last_val, "frames_seen": frames_seen}
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[eval] step {step}: val_loss={last_val:.5f} (best {min(best_val, last_val):.5f})")
        if last_val < best_val - 1e-9:
            best_val, best_step, bad_evals = last_val, step, 0
            _save_atomic({"model": model.state_dict(), "model_config": _model_config(cfg), "config": cfg,
                          "norm_mean": mean.cpu(), "norm_std": std.cpu(), "step": step, "val_loss": last_val},
                         out_dir / "best.pt")
        else:
            bad_evals += 1
        write_results(False)

    try:
        while step < t["max_steps"]:
            lr = _lr_at(step, t)
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(t["grad_accum_steps"]):
                x, y, w = _prepare(train_c, train_c.sample_starts(rng, t["batch_size"]), mean, std, part_map, device)
                with torch.autocast(**autocast_kwargs):
                    pred = model(x)
                se, ws = _masked_mse_sums(pred.float(), y, w)
                loss = se / ws.clamp_min(1.0) / t["grad_accum_steps"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}. Try a lower train.learning_rate.")
                scaler.scale(loss).backward()
                loss_sum += float(loss)
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])) if t["grad_clip_norm"] else 0.0
            scaler.step(opt)
            scaler.update()
            step += 1
            frames_seen += frames_per_step
            last_train_loss = loss_sum

            if step % t["log_every"] == 0 or step == 1:
                rec = {"event": "train", "step": step, "train_loss": loss_sum, "lr": lr, "grad_norm": gnorm,
                       "frames_seen": frames_seen, "minutes": round((time.time() - start_time) / 60.0, 2)}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                print(f"[train] step {step}/{t['max_steps']} loss={loss_sum:.5f} lr={lr:.2e} gnorm={gnorm:.2f}")

            if step % t["eval_every"] == 0:
                do_eval()
                if t["patience_evals"] and bad_evals >= t["patience_evals"]:
                    stop_reason = "early_stop"
                    break
            if step % t["save_every"] == 0:
                checkpoint(out_dir / "last.pt")
            if r["max_train_minutes"] and (time.time() - start_time) / 60.0 >= r["max_train_minutes"]:
                stop_reason = "time_limit"
                break
        if last_eval_step != step:
            do_eval()
        checkpoint(out_dir / "last.pt")
        res = write_results(stop_reason != "time_limit")
    finally:
        log_f.close()
    print(f"[done] {stop_reason}: best val {res['best_val_loss']} at step {res['best_step']} "
          f"({res['n_params_millions']}M params). Results: {out_dir / 'results.json'}")
    return res


def main(argv: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    ap = argparse.ArgumentParser(description="Train MoCapGPT from a YAML config.")
    ap.add_argument("--config", help="YAML config file (see configs/gpt_sample.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="section.key=value", help="override config values")
    ap.add_argument("--count_params", action="store_true", help="print the parameter count and exit")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.count_params:
        n = count_params(cfg)
        print(f"{n:,} parameters ({n / 1e6:.2f}M)")
        return None
    return run(cfg)


if __name__ == "__main__":
    main()
