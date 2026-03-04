#!/usr/bin/env python3
"""
Plot a graph for each loss metric (G_A, cycle_A, cycle_B, idt_A, D_A) from all three rounds.
Uses the same data as loss_log_summary.py. Saves one PNG per metric in important_metrics/.

Run from project root: python scripts/plot_loss_metrics.py
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse loading from loss_log_summary
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_log_summary import load_round, REPORT_SPEC

ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = ROOT / "important_metrics"

# Slide color scheme: round1, round2, round3
COLORS = ["#E1932C", "#3D9E71", "#133968"]
EDGE = "#F0DF87"


def by_epoch(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Average metric per epoch. Returns (epochs, values)."""
    by_epoch_d = defaultdict(list)
    for r in rows:
        if key not in r:
            continue
        by_epoch_d[r["epoch"]].append(r[key])
    epochs = sorted(by_epoch_d.keys())
    vals = [sum(by_epoch_d[e]) / len(by_epoch_d[e]) for e in epochs]
    return epochs, vals


def plot_metric(metric_key: str, label: str, all_rows: dict, ax: plt.Axes) -> None:
    """Draw one metric on given axes: 3 lines (round1, round2, round3) vs epoch."""
    for i, (rnd, rows) in enumerate(all_rows.items()):
        if not rows or metric_key not in rows[0]:
            continue
        epochs, vals = by_epoch(rows, metric_key)
        if not epochs:
            continue
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, vals, color=color, linewidth=1.2, label=rnd, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def main():
    all_rows = {}
    for rnd in ["round1", "round2", "round3"]:
        rows = load_round(rnd)
        all_rows[rnd] = rows
        if rows:
            print(f"  {rnd}: {len(rows)} entries")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # One combined figure with all metrics (2x2 = 4 panels; we have 5, so use 2x3)
    n_metrics = len(REPORT_SPEC)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, (metric_key, label, _typical, _interp) in enumerate(REPORT_SPEC):
        if idx < len(axes):
            plot_metric(metric_key, label, all_rows, axes[idx])
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Loss metrics over training (all rounds)", fontsize=14, y=1.02)
    fig.tight_layout()
    combined_path = METRICS_DIR / "graph_all_loss_metrics.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", combined_path)

    # Optional: also save individual figures
    for metric_key, label, _typical, _interp in REPORT_SPEC:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_metric(metric_key, label, all_rows, ax)
        ax.set_title(f"{label} over training (all rounds)", fontsize=14)
        fig.tight_layout()
        out_path = METRICS_DIR / f"graph_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved", out_path)

    print("Done. Graphs in", METRICS_DIR)


if __name__ == "__main__":
    main()
