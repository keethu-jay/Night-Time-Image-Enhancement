#!/usr/bin/env python3
"""
Create well-labeled individual histograms for MSE, SSIM, and Identity Fidelity
using the slide color scheme. Reads from the CSV files in important_metrics/.

Run from project root: python scripts/plot_individual_histograms.py
Output: important_metrics/histogram_mse.png, histogram_ssim.png, histogram_identity_fidelity.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = ROOT / "important_metrics"

# Slide color scheme (hex)
COLORS = [
    "#E1932C",  # orange
    "#F0DF87",  # light yellow
    "#A0CED9",  # light blue
    "#3D9E71",  # green
    "#133968",  # dark blue
]


def load_csv_column(path: Path, col_name: str) -> list[float]:
    if not path.exists():
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append(float(row[col_name]))
            except (KeyError, ValueError):
                pass
    return out


def plot_one_histogram(
    values: list[float],
    out_path: Path,
    title: str,
    xlabel: str,
    bar_color: str,
    edge_color: str = "#F0DF87",
    n_bins: int = 25,
) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    bins = min(n_bins, max(1, n // 5))
    ax.hist(values, bins=bins, color=bar_color, edgecolor=edge_color, linewidth=0.8)
    ax.axvline(mean_val, color="#133968", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.4f}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title(f"{title}\n(N = {n})", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def main():
    mses = load_csv_column(METRICS_DIR / "histogram_mse_night_to_day.csv", "mse")
    ssims = load_csv_column(METRICS_DIR / "histogram_ssim_night_to_day.csv", "ssim")
    idt_fids = load_csv_column(METRICS_DIR / "histogram_identity_fidelity_day_to_day.csv", "identity_fidelity")

    # MSE: dark blue bars, light yellow edges
    plot_one_histogram(
        mses,
        METRICS_DIR / "histogram_mse.png",
        title="MSE distribution (Night-to-Day translations)",
        xlabel="MSE (lower is better)",
        bar_color=COLORS[4],
        edge_color=COLORS[1],
    )

    # SSIM: green bars, light yellow edges
    plot_one_histogram(
        ssims,
        METRICS_DIR / "histogram_ssim.png",
        title="SSIM distribution (Night-to-Day translations)",
        xlabel="SSIM (higher is better)",
        bar_color=COLORS[3],
        edge_color=COLORS[1],
    )

    # Identity Fidelity: orange bars, light blue edges
    plot_one_histogram(
        idt_fids,
        METRICS_DIR / "histogram_identity_fidelity.png",
        title="Identity Fidelity distribution (Day-to-Day)",
        xlabel="Identity Fidelity (1 - L1; higher is better)",
        bar_color=COLORS[0],
        edge_color=COLORS[2],
    )

    print("Done. Individual histograms use slide colors: #E1932C, #F0DF87, #A0CED9, #3D9E71, #133968.")


if __name__ == "__main__":
    main()
