"""Quick scaling table/chart straight from results.json files — no dataset
or SMPL-X model needed, since it only reads numbers each run already saved.

Plots parameter count against the run's own best validation loss. That
loss is only safely comparable between runs that used the same data.dir
and seed (so they share the same feature normalization) — true for a set
of sizes trained from the same sample config, which is the normal case
here. It's also in normalized rotation/velocity feature units, not a
physically meaningful number — for a millimeter accuracy metric that's
comparable across different setups too, use evaluate_scaling.py instead
(that one needs the dataset, since it reruns the model on real frames).

Usage:
    python -m src.gpt.scaling_curve_from_results runs/gpt_xs runs/gpt_s runs/gpt_m runs/gpt_l runs/gpt_xl
    python -m src.gpt.scaling_curve_from_results runs/* --out_dir report
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_row(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — is this a finished (or at least checkpointed) run directory?")
    r = json.loads(path.read_text())
    if r.get("best_val_loss") is None:
        raise ValueError(f"{path} has no best_val_loss yet (no validation has completed for this run).")
    return r


def write_markdown_table(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["n_params"])
    lines = ["| Model | Parameters | Best val loss | Steps | Status |", "|---|---|---|---|---|"]
    for r in rows:
        status = "finished" if r["finished"] else "stopped early"
        lines.append(f"| {r['name']} | {r['n_params_millions']:.2f}M | {r['best_val_loss']:.4f} | "
                     f"{r['steps_done']} | {status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scaling_chart(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["n_params"])
    x = [r["n_params"] for r in rows]
    y = [r["best_val_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.plot(x, y, color="#2F5496", linewidth=2, marker="o", markersize=7,
            markerfacecolor="#2F5496", markeredgecolor="#FFFFFF", markeredgewidth=1, zorder=3)
    for r in rows:
        ax.annotate(r["name"], (r["n_params"], r["best_val_loss"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", color="#333333")
    ax.set_ylabel("Best validation loss (normalized units, lower is better)", color="#333333")
    ax.set_title("Model size vs. validation loss", color="#1A1A1A", fontsize=11)
    ax.grid(True, which="both", linewidth=0.6, color="#DDDDDD", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BBBBBB")
    ax.tick_params(colors="#555555")
    fig.tight_layout()
    fig.savefig(path, facecolor="#FFFFFF")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out_dir", default="scaling_report_quick")
    args = ap.parse_args(argv)

    expanded: List[Path] = []
    for pattern in args.run_dirs:
        matches = [Path(p) for p in glob.glob(pattern)] or [Path(pattern)]
        expanded.extend(p for p in matches if p.is_dir())
    if not expanded:
        raise FileNotFoundError(f"No directories matched: {args.run_dirs}")

    rows = [load_row(d) for d in expanded]
    out_dir = Path(args.out_dir)
    write_markdown_table(rows, out_dir / "scaling_table.md")
    write_scaling_chart(rows, out_dir / "scaling_chart.png")
    print(f"[done] wrote {out_dir / 'scaling_table.md'} and {out_dir / 'scaling_chart.png'}")
    return rows


if __name__ == "__main__":
    main()
