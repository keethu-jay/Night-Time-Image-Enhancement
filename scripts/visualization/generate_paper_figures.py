#!/usr/bin/env python3
"""
Generate all figures and Table 1 for the paper (CVPR/CS research standards).

Produces:
  Figure 1: CycleGAN architecture (G, F, discriminators, cycle consistency, identity).
  Table 1:  Summary of peak metric performance (Round, Epoch, MSE, SSIM, Identity Fidelity).
  Figure 2: 2x2 loss gallery (G_A, D_A, Cycle Loss, Identity Loss).
  Figure 3: Round 2 vs Round 3 qualitative (Night | Round2 output | Round3 output).
  Figure 4: Generalization histograms (MSE, SSIM, Identity Fidelity, N=440).
  Figure 5: Adversarial gap (zoomed end of Round 3, G_A and D_A).
  Figure 6: Traditional (CLAHE) vs CycleGAN comparison.

Run from project root: python scripts/visualization/generate_paper_figures.py
Outputs: paper_figures/ (or --out-dir) with figure_1.png ... figure_6.png and table_1_performance.md
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "paper_figures"
CHECKPOINTS = ROOT / "checkpoints"
LOSS_LOGS = CHECKPOINTS / "loss_logs"
RESULTS = ROOT / "results"
METRICS_DIR = ROOT / "important_metrics"
SIZE = (256, 256)

# Best epochs (Round 2 peaks)
BEST_MSE_EPOCH = "55"
BEST_SSIM_EPOCH = "70"
BEST_IDT_EPOCH = "140"
ROUND2 = "round2"
ROUND3 = "round3"

# Loss log parsing (minimal copy to keep script self-contained)
_LOSS_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*(\d+).*?\)\s*,"
    r"\s*D_A:\s*([\d.]+),\s*G_A:\s*([\d.]+),\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*([\d.]+),\s*G_B:\s*([\d.]+),\s*cycle_B:\s*([\d.]+),\s*idt_B:\s*([\d.]+)"
)
CSV_COLS = ["epoch", "iteration", "D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]


def _norm_key(k: str) -> str:
    m = {"cycle_a": "cycle_A", "idt_a": "idt_A", "g_a": "G_A", "d_a": "D_A"}
    return m.get(k.lower().replace(" ", ""), k)


def parse_txt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        {
            "epoch": int(m.group(1)), "iteration": int(m.group(2)),
            "D_A": float(m.group(3)), "G_A": float(m.group(4)),
            "cycle_A": float(m.group(5)), "idt_A": float(m.group(6)),
            "D_B": float(m.group(7)), "G_B": float(m.group(8)),
            "cycle_B": float(m.group(9)), "idt_B": float(m.group(10)),
        }
        for m in _LOSS_RE.finditer(text)
    ]


def parse_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        key_map = {fn: _norm_key(fn) for fn in (r.fieldnames or [])}
        for line in r:
            try:
                row = {}
                for fn, val in line.items():
                    std = key_map.get(fn, fn)
                    if std in CSV_COLS:
                        row[std] = int(val) if std in ("epoch", "iteration") else float(val)
                if len(row) >= 8:
                    rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def load_round(round_name: str) -> list[dict]:
    base = f"loss_log_{round_name}"
    csv_path = LOSS_LOGS / f"{base}.csv"
    txt_path = LOSS_LOGS / f"{base}.txt"
    if csv_path.exists():
        r = parse_csv(csv_path)
        if r:
            return r
    if txt_path.exists():
        r = parse_txt(txt_path)
        if r:
            return r
    if round_name == "round3":
        generic = LOSS_LOGS / "loss_log.txt"
        if generic.exists():
            return parse_txt(generic)
    return []


def by_epoch(rows: list[dict], key: str):
    from collections import defaultdict
    by_epoch_d = defaultdict(list)
    for r in rows:
        if key not in r:
            continue
        by_epoch_d[r["epoch"]].append(r[key])
    epochs = sorted(by_epoch_d.keys())
    vals = [sum(by_epoch_d[e]) / len(by_epoch_d[e]) for e in epochs]
    return epochs, vals


def figure_1_architecture(out_path: Path) -> None:
    """Figure 1: Cycle-Consistent Adversarial Network (CycleGAN) Pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Boxes: Night (left), Day (right), G and F in between
    # N at (1,3), D at (9,3). G: N->D at (4,4), F: D->N at (6,2). D_A near D, D_B near N.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.transforms import Bbox

    def box(x, y, w, h, label, facecolor="lightblue", edgecolor="black"):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(p)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold")

    box(1.5, 3, 1.2, 0.8, "Night\n(N)", facecolor="#E8F4F8")
    box(8.5, 3, 1.2, 0.8, "Day\n(D)", facecolor="#FFF4E6")
    box(4, 4.2, 1.4, 0.6, r"$G: N \to D$", facecolor="#D4EDDA")
    box(6, 1.8, 1.4, 0.6, r"$F: D \to N$", facecolor="#D4EDDA")
    box(8.5, 4.8, 1.0, 0.5, r"$D_A$", facecolor="#F8D7DA")
    box(1.5, 1.2, 1.0, 0.5, r"$D_B$", facecolor="#F8D7DA")

    # Arrows: cycle N -> G -> D -> F -> N
    ax.annotate("", xy=(2.1, 3.5), xytext=(3.3, 4.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(7.9, 3.5), xytext=(4.7, 4.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(7.9, 2.5), xytext=(7.9, 2.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(2.1, 2.5), xytext=(5.3, 1.8), arrowprops=dict(arrowstyle="->", lw=2))
    # Identity shortcuts
    ax.annotate("", xy=(1.5, 3.6), xytext=(1.5, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="gray", linestyle="--"))
    ax.annotate("", xy=(8.5, 3.6), xytext=(8.5, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="gray", linestyle="--"))

    ax.text(2.5, 3.8, "Cycle", fontsize=8)
    ax.text(5, 3, "Cycle", fontsize=8)
    ax.text(7, 2.2, "Cycle", fontsize=8)
    ax.text(3.5, 2.2, "Cycle", fontsize=8)
    ax.text(1.5, 4.0, "Idt", fontsize=8, color="gray")
    ax.text(8.5, 4.0, "Idt", fontsize=8, color="gray")

    ax.text(5, 5.5, "Figure 1: Cycle-Consistent Adversarial Network (CycleGAN) Pipeline.", fontsize=12, fontweight="bold")
    ax.text(5, 0.4, "Cycle consistency: N→G→D→F→N; Identity: G(D)≈D, F(N)≈N.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def table_1_performance(out_path: Path) -> None:
    """Table 1: Summary of Peak Metric Performance across Training Phases."""
    # Use known Round 2 peaks and batch Identity Fidelity from dashboard/model_selection
    rows = [
        ("Best MSE", "2", BEST_MSE_EPOCH, "0.0615", "0.292", "98.0"),
        ("Best SSIM", "2", BEST_SSIM_EPOCH, "0.0973", "0.317", "98.0"),
        ("Best Identity", "2", BEST_IDT_EPOCH, "0.0865", "0.253", "100.0"),
    ]
    leaderboard = METRICS_DIR / "batch_leaderboard.csv"
    if leaderboard.exists():
        try:
            with open(leaderboard, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    model = (row.get("model") or "").lower()
                    mse = row.get("avg_mse") or row.get("mse") or "—"
                    ssim = row.get("avg_ssim") or row.get("ssim") or "—"
                    idt = row.get("identity_fidelity") or row.get("idt") or "—"
                    if "mse" in model and "ssim" not in model:
                        rows[0] = ("Best MSE", "2", BEST_MSE_EPOCH, str(mse), str(ssim), str(idt))
                    elif "ssim" in model:
                        rows[1] = ("Best SSIM", "2", BEST_SSIM_EPOCH, str(mse), str(ssim), str(idt))
                    elif "idt" in model or "identity" in model:
                        rows[2] = ("Best Identity", "2", BEST_IDT_EPOCH, str(mse), str(ssim), str(idt))
        except Exception:
            pass

    md = [
        "# Table 1: Summary of Peak Metric Performance across Training Phases",
        "",
        "| Model | Round | Epoch | MSE | SSIM | Identity Fidelity (%) |",
        "|-------|-------|-------|-----|------|------------------------|",
    ]
    for model, rnd, ep, mse, ssim, idt in rows:
        md.append(f"| {model} | {rnd} | {ep} | {mse} | {ssim} | {idt} |")
    md.append("")
    md.append("(Peak metrics from Round 2; Identity Fidelity from batch N=440 Day→Day.)")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print("Saved", out_path)


def figure_2_loss_gallery(out_path: Path) -> None:
    """Figure 2: Adversarial and Structural Loss Convergence Profiles (2x2)."""
    all_rows = {}
    for rnd in ["round1", "round2", "round3"]:
        all_rows[rnd] = load_round(rnd)

    panels = [
        ("G_A", r"$G_A$ Loss (adversarial)"),
        ("D_A", r"$D_A$ Loss (discriminator)"),
        ("cycle_A", "Cycle Loss (N→D→N)"),
        ("idt_A", "Identity Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colors = ["#E1932C", "#3D9E71", "#133968"]
    for idx, (key, title) in enumerate(panels):
        ax = axes[idx]
        for i, (rnd, rows) in enumerate(all_rows.items()):
            if not rows or key not in (rows[0] or {}):
                continue
            epochs, vals = by_epoch(rows, key)
            if not epochs:
                continue
            ax.plot(epochs, vals, color=colors[i % 3], linewidth=1.2, label=rnd, alpha=0.9)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Figure 2: Adversarial and Structural Loss Convergence Profiles.", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img) / 255.0


def figure_3_round2_vs_round3(out_path: Path) -> None:
    """Figure 3: Impact of Inference-Time Refinement (Night | Round2 | Round3)."""
    prof_night = ROOT / "night-to-day-translation" / "data" / "raw" / "professor_pair" / "night.jpg"
    test_a = ROOT / "datasets" / "night2day" / "testA"
    if not prof_night.exists() and test_a.exists():
        first = next((f for f in sorted(test_a.glob("*.png")) or sorted(test_a.glob("*.jpg"))), None)
        night_path = first or prof_night
    else:
        night_path = prof_night
    if not night_path.exists():
        print("Skip Figure 3: no night image at", night_path)
        return

    r2_dir = RESULTS / ROUND2 / f"test_{BEST_MSE_EPOCH}" / "images"
    r3_dir = RESULTS / ROUND3 / "test_100" / "images"
    if not r3_dir.exists():
        r3_dir = RESULTS / ROUND3 / "test_latest" / "images"
    fake_r2 = None
    if r2_dir.exists():
        for f in ["grading_night_fake_B.png", "night_fake_B.png"]:
            if (r2_dir / f).exists():
                fake_r2 = (r2_dir / f).resolve()
                break
        if fake_r2 is None:
            fakes = list(r2_dir.glob("*_fake*.png"))
            fake_r2 = fakes[0] if fakes else None
    fake_r3 = None
    if r3_dir.exists():
        for f in ["grading_night_fake_B.png", "night_fake_B.png"]:
            if (r3_dir / f).exists():
                fake_r3 = (r3_dir / f).resolve()
                break
        if fake_r3 is None:
            fakes = list(r3_dir.glob("*_fake*.png"))
            fake_r3 = fakes[0] if fakes else None

    ims = [load_image(night_path)]
    if fake_r2 and fake_r2.exists():
        ims.append(load_image(fake_r2))
    else:
        ims.append(np.zeros((*SIZE[::-1], 3)))
    if fake_r3 and fake_r3.exists():
        ims.append(load_image(fake_r3))
    else:
        ims.append(np.zeros((*SIZE[::-1], 3)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    labels = ["Original Night", "Round 2 (best MSE)", "Round 3 (refined)"]
    for ax, im, label in zip(axes, ims, labels):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.suptitle("Figure 3: Impact of Inference-Time Refinement on Visual Integrity.", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def figure_4_histograms(out_path: Path) -> None:
    """Figure 4: Statistical Distribution of Performance Metrics (N=440)."""
    files = [
        (METRICS_DIR / "histogram_mse_night_to_day.csv", "mse", "MSE (Night→Day)", "MSE"),
        (METRICS_DIR / "histogram_ssim_night_to_day.csv", "ssim", "SSIM (Night→Day)", "SSIM"),
        (METRICS_DIR / "histogram_identity_fidelity_day_to_day.csv", "identity_fidelity", "Identity Fidelity (Day→Day)", "Identity Fidelity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (csv_path, col, title, xlabel) in zip(axes, files):
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"No data: {csv_path.name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        vals = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    v = float(row.get(col, row.get(list(row.keys())[-1])))
                    vals.append(v)
                except (ValueError, KeyError, IndexError):
                    continue
        if vals:
            ax.hist(vals, bins=min(40, len(vals)//3 or 10), color="#3D9E71", edgecolor="white", alpha=0.8)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Figure 4: Statistical Distribution of Performance Metrics (N=440).", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def figure_5_adversarial_gap(out_path: Path) -> None:
    """Figure 5: Evidence of Incomplete Adversarial Convergence (end of Round 3, G_A and D_A)."""
    rows = load_round("round3")
    if not rows:
        print("Skip Figure 5: no round3 loss log")
        return
    frac = 0.25
    n = max(1, int(len(rows) * frac))
    tail = rows[-n:]
    indices = list(range(len(tail)))
    vals_ga = [r.get("G_A") for r in tail]
    vals_da = [r.get("D_A") for r in tail]
    vals_ga = [v for v in vals_ga if v is not None]
    vals_da = [v for v in vals_da if v is not None]
    if not vals_ga or not vals_da:
        print("Skip Figure 5: no G_A/D_A in round3 log")
        return
    x = indices[: len(vals_ga)]
    if len(vals_da) < len(x):
        x = indices[: len(vals_da)]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(x, vals_ga[: len(x)], color="#133968", linewidth=1.5, label=r"$G_A$")
    axes[0].set_ylabel(r"$G_A$ Loss", fontsize=10)
    axes[0].set_title(r"Generator $G_A$ (last 25% of Round 3)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, vals_da[: len(x)], color="#E1932C", linewidth=1.5, label=r"$D_A$")
    axes[1].set_xlabel("Iteration (tail of Round 3)", fontsize=10)
    axes[1].set_ylabel(r"$D_A$ Loss", fontsize=10)
    axes[1].set_title(r"Discriminator $D_A$ (last 25% of Round 3)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Figure 5: Evidence of Incomplete Adversarial Convergence.", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def figure_6_traditional_vs_gan(out_path: Path) -> None:
    """Figure 6: Traditional Enhancement (CLAHE) vs Proposed CycleGAN."""
    try:
        import cv2
    except ImportError:
        print("Skip Figure 6: opencv not installed")
        return
    night_path = ROOT / "night-to-day-translation" / "data" / "raw" / "professor_pair" / "night.jpg"
    if not night_path.exists():
        test_a = ROOT / "datasets" / "night2day" / "testA"
        if test_a.exists():
            first = next((f for f in sorted(test_a.glob("*.png")) or sorted(test_a.glob("*.jpg"))), None)
            night_path = first or night_path
    if not night_path.exists():
        print("Skip Figure 6: no night image")
        return

    rgb = np.array(Image.open(night_path).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)) / 255.0
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    clahe_rgb = np.clip(cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0, 0, 1)

    fake_path = RESULTS / ROUND2 / f"test_{BEST_MSE_EPOCH}" / "images"
    if not fake_path.exists():
        fake_path = RESULTS / ROUND3 / "test_100" / "images"
    if not fake_path.exists():
        fake_path = RESULTS / ROUND3 / "test_latest" / "images"
    fakes = list(fake_path.glob("*_fake*.png")) if fake_path.exists() else []
    if fakes:
        gan_im = load_image(fakes[0])
    else:
        gan_im = np.zeros((*SIZE[::-1], 3))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("Input (Night)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(clahe_rgb)
    axes[1].set_title("CLAHE (traditional)", fontsize=11)
    axes[1].axis("off")
    axes[2].imshow(np.clip(gan_im, 0, 1))
    axes[2].set_title("Proposed (CycleGAN)", fontsize=11)
    axes[2].axis("off")
    fig.suptitle("Figure 6: Traditional Enhancement vs Proposed Deep Learning Pipeline.", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def main():
    ap = argparse.ArgumentParser(description="Generate all paper figures and Table 1.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for figures and table.")
    ap.add_argument("--skip", nargs="*", default=[], help="Skip figures by number, e.g. --skip 1 5")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    if "1" not in skip:
        figure_1_architecture(out_dir / "figure_1_architecture.png")
    table_1_performance(out_dir / "table_1_performance.md")
    if "2" not in skip:
        figure_2_loss_gallery(out_dir / "figure_2_loss_gallery.png")
    if "3" not in skip:
        figure_3_round2_vs_round3(out_dir / "figure_3_round2_vs_round3.png")
    if "4" not in skip:
        figure_4_histograms(out_dir / "figure_4_histograms.png")
    if "5" not in skip:
        figure_5_adversarial_gap(out_dir / "figure_5_adversarial_gap.png")
    if "6" not in skip:
        figure_6_traditional_vs_gan(out_dir / "figure_6_traditional_vs_gan.png")

    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
