"""Turn a set of finished training runs (one per model size) into a scaling
table and chart: parameter count vs. a real-unit accuracy metric.

Training loss is not that metric. Its value depends on the per-feature
normalization computed from whatever data a given run happened to load, so
it isn't safely comparable between two runs, and it's in normalized 6D
rotation / velocity units that mean nothing physically. Mean Per-Joint
Position Error (MPJPE, in millimeters) is: predicted and true pose are each
run through the real SMPL-X forward kinematics (via `SMPLXBody`), and the
resulting joint positions are compared directly. It's the standard metric in
the human motion literature, so it's also comparable to numbers other work
reports.

Two things this MPJPE does NOT cover, on purpose:
  * Body shape (betas) isn't in the 315-d model feature vector the GPT
    predicts, so both prediction and ground truth are run through the body
    model with betas=0 (a fixed neutral shape). Since both sides use the
    same shape, this is still a fair comparison of *pose* accuracy across
    model sizes, but the absolute mm values would look different against
    a real subject's actual body shape.
  * Translation likewise isn't compared: both are placed at the same
    (zero) root position before forward kinematics, so this is a
    root-relative pose error, not a measure of whether the model tracks a
    body's path through space.
  * This measures single-step (teacher-forced) next-frame prediction, the
    same task the model was trained on — not autoregressive rollout, which
    compounds error over many steps and is a separate, harder evaluation.

Usage:
    python -m src.gpt.evaluate_scaling runs/gpt_xs runs/gpt_s runs/gpt_m runs/gpt_l runs/gpt_xl
    python -m src.gpt.evaluate_scaling runs/* --out_dir report --max_windows 300
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.gpt.gpt import MoCapGPT
from src.gpt.train import MotionCorpus, build_corpora
from src.smplx.features import MODEL_FEATURE_DIM, model_features_to_canonical
from src.smplx.fitting.body import BodyParams, SMPLXBody


def _zero_trans_canonical(features_unnorm: np.ndarray) -> np.ndarray:
    """(T, 315) unnormalized model features -> (T, 159) canonical pose with
    translation forced to zero (see module docstring: root-relative only)."""
    canonical = model_features_to_canonical(features_unnorm, initial_trans=np.zeros(3))
    canonical[:, 0:3] = 0.0
    return canonical


def _forward_joints(body: SMPLXBody, canonical: np.ndarray) -> np.ndarray:
    t = canonical.shape[0]
    params = BodyParams(
        betas=torch.zeros(10, dtype=torch.float64),
        global_orient=torch.tensor(canonical[:, 3:6]),
        body_pose=torch.tensor(canonical[:, 6:69]),
        left_hand_pose=torch.tensor(canonical[:, 69:114]),
        right_hand_pose=torch.tensor(canonical[:, 114:159]),
        transl=torch.tensor(canonical[:, 0:3]),
    )
    with torch.no_grad():
        return body.forward(params).joints.numpy()  # (T, J, 3), meters


@torch.no_grad()
def compute_mpjpe_mm(
    model: MoCapGPT, val_corpus: MotionCorpus, mean: torch.Tensor, std: torch.Tensor,
    body: SMPLXBody, device: torch.device, batch_size: int = 16, max_windows: Optional[int] = 200,
) -> Tuple[float, int]:
    """Mean per-joint position error, in millimeters, over the validation
    set's fixed (non-overlapping) windows, on single-step next-frame
    prediction. Returns (mpjpe_mm, number of joint-frames it was averaged over)."""
    model.eval()
    starts = val_corpus.fixed_starts
    if len(starts) == 0:
        raise ValueError("Validation set has no full window to evaluate.")
    if max_windows and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]

    dist_sum, n = 0.0, 0
    for i in range(0, len(starts), batch_size):
        frames, _mask = val_corpus.gather(starts[i:i + batch_size])
        frames = frames.to(device)
        x_norm = (frames[:, :-1] - mean) / std
        pred_norm = model(x_norm).float()
        pred_unnorm = (pred_norm * std + mean).cpu().numpy()
        true_unnorm = frames[:, 1:].cpu().numpy()

        for b in range(pred_unnorm.shape[0]):
            joints_pred = _forward_joints(body, _zero_trans_canonical(pred_unnorm[b]))
            joints_true = _forward_joints(body, _zero_trans_canonical(true_unnorm[b]))
            dist = np.linalg.norm(joints_pred - joints_true, axis=-1)  # (T, J), meters
            dist_sum += float(dist.sum())
            n += dist.size

    return dist_sum / max(n, 1) * 1000.0, n


def _load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MoCapGPT(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    mean, std = ckpt["norm_mean"].to(device), ckpt["norm_std"].to(device)
    return model, ckpt["config"], mean, std


def evaluate_run(run_dir: Path, body: SMPLXBody, device: torch.device,
                  batch_size: int = 16, max_windows: Optional[int] = 200,
                  data_dir: Optional[str] = None) -> Dict[str, Any]:
    """`data_dir`: override the `data.dir` each run's own saved config
    points at. Needed whenever evaluation runs somewhere other than where
    training ran — most commonly training on Kaggle (`data.dir=/kaggle/
    input/...`) and evaluating on a different machine after downloading
    the run folders, where that path doesn't exist. Point it at a local
    copy of the *same* dataset; the validation split is reproduced from
    the run's own seed and file ordering, so it only matches if the file
    set is the same."""
    results_path, ckpt_path = run_dir / "results.json", run_dir / "best.pt"
    if not results_path.is_file():
        raise FileNotFoundError(f"{results_path} not found — is this a finished run directory?")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{ckpt_path} not found — is this a finished run directory?")

    results = json.loads(results_path.read_text())
    model, cfg, mean, std = _load_checkpoint(ckpt_path, device)
    if data_dir:
        cfg = copy.deepcopy(cfg)
        cfg["data"]["dir"] = data_dir
    _train_c, val_c, _info = build_corpora(cfg)  # same seed + data.dir -> the same split used in training
    mpjpe_mm, n_eval = compute_mpjpe_mm(model, val_c, mean, std, body, device, batch_size, max_windows)

    return {
        "name": results["name"],
        "n_params": results["n_params"],
        "n_params_millions": results["n_params_millions"],
        "best_val_loss": results["best_val_loss"],
        "baseline_val_loss_mean": results.get("baseline_val_loss_mean"),
        "mpjpe_mm": mpjpe_mm,
        "n_eval_joint_frames": n_eval,
        "steps_done": results["steps_done"],
        "finished": results["finished"],
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Table + chart
# ---------------------------------------------------------------------------
def write_markdown_table(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["n_params"])
    lines = [
        "| Model | Parameters | MPJPE (mm) | Best val loss | Steps | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = "finished" if r["finished"] else "stopped early"
        lines.append(
            f"| {r['name']} | {r['n_params_millions']:.2f}M | {r['mpjpe_mm']:.1f} | "
            f"{r['best_val_loss']:.4f} | {r['steps_done']} | {status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scaling_chart(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["n_params"])
    x = [r["n_params"] for r in rows]
    y = [r["mpjpe_mm"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    ax.plot(x, y, color="#2F5496", linewidth=2, marker="o", markersize=7,
            markerfacecolor="#2F5496", markeredgecolor="#FFFFFF", markeredgewidth=1, zorder=3)
    for r in rows:
        ax.annotate(r["name"], (r["n_params"], r["mpjpe_mm"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", color="#333333")
    ax.set_ylabel("MPJPE (mm, lower is better)", color="#333333")
    ax.set_title("Model size vs. pose accuracy on held-out validation clips", color="#1A1A1A", fontsize=11)
    ax.grid(True, which="both", linewidth=0.6, color="#DDDDDD", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BBBBBB")
    ax.tick_params(colors="#555555")

    fig.tight_layout()
    fig.savefig(path, facecolor="#FFFFFF")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="one or more finished run directories (each holding results.json and best.pt)")
    ap.add_argument("--out_dir", default="scaling_report")
    ap.add_argument("--max_windows", type=int, default=200, help="validation windows per run; 0 = use all of them")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default=None,
                    help="override the data.dir saved in each run's config, e.g. when evaluating "
                         "on a machine other than where training ran (see evaluate_run's docstring)")
    args = ap.parse_args(argv)

    expanded: List[Path] = []
    for pattern in args.run_dirs:
        matches = [Path(p) for p in glob.glob(pattern)] or [Path(pattern)]
        expanded.extend(p for p in matches if p.is_dir())
    if not expanded:
        raise FileNotFoundError(f"No directories matched: {args.run_dirs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = SMPLXBody(device="cpu")  # forward kinematics here is a small, occasional eval step, not training

    rows = []
    for run_dir in expanded:
        print(f"[eval] {run_dir}")
        rows.append(evaluate_run(run_dir, body, device, args.batch_size, args.max_windows or None, args.data_dir))
        print(f"       {rows[-1]['n_params_millions']:.2f}M params -> MPJPE {rows[-1]['mpjpe_mm']:.1f} mm")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown_table(rows, out_dir / "scaling_table.md")
    write_scaling_chart(rows, out_dir / "scaling_chart.png")
    print(f"[done] wrote {out_dir / 'scaling_table.md'} and {out_dir / 'scaling_chart.png'}")
    return rows


if __name__ == "__main__":
    main()
