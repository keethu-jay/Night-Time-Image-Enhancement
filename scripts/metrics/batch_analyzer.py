#!/usr/bin/env python3
"""
Step 2: Automated Metric Comparison across the three CycleGAN result folders.

Iterates through 220 night-to-day results (avg MSE, avg SSIM) and 220 day-to-day results
(Identity Fidelity = 1 - L1) for model_best_mse, model_best_ssim, model_best_idt.
Outputs a Leaderboard Table for slides.

Run from project root: python scripts/batch_analyzer.py

Optional: --results_dir, --dataroot (large_test path), --epochs to match test_<epoch> folders.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
DATAROOT = ROOT / "datasets" / "large_test"
SIZE = (256, 256)

# Model names and expected epoch subfolder (test_55, test_70, test_140)
MODELS = [
    ("model_best_mse", "55"),
    ("model_best_ssim", "70"),
    ("model_best_idt", "140"),
]


def load_rgb(path: Path) -> np.ndarray:
    """Load image, resize to SIZE, return RGB float [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64) / 255.0


def load_grayscale(path: Path) -> np.ndarray:
    """Load image, resize to SIZE, grayscale float [0,1]."""
    arr = load_rgb(path)
    return np.mean(arr[..., :3], axis=-1)


def mse_channelwise_avg(real: np.ndarray, fake: np.ndarray) -> float:
    """Average channel-wise MSE (R,G,B)."""
    mse_r = float(np.mean((real[..., 0] - fake[..., 0]) ** 2))
    mse_g = float(np.mean((real[..., 1] - fake[..., 1]) ** 2))
    mse_b = float(np.mean((real[..., 2] - fake[..., 2]) ** 2))
    return (mse_r + mse_g + mse_b) / 3.0


def ssim_one(real: np.ndarray, fake: np.ndarray) -> float:
    """SSIM on grayscale; data range 1.0."""
    if ssim_func is None:
        return float("nan")
    r = np.mean(real, axis=-1) if real.ndim == 3 else real
    f = np.mean(fake, axis=-1) if fake.ndim == 3 else fake
    return float(ssim_func(r, f, data_range=1.0))


def l1_one(real: np.ndarray, fake: np.ndarray) -> float:
    """Mean absolute difference (L1) over pixels and channels."""
    return float(np.mean(np.abs(real.astype(np.float64) - fake.astype(np.float64))))


def collect_fake_paths(images_dir: Path, pattern: str = "*_fake*.png") -> list[Path]:
    """Sorted list of fake image paths; prefer *_fake.png then *_fake_B.png."""
    paths = list(images_dir.glob(pattern))
    # Sort by stem so 000_fake.png, 001_fake.png, ...
    paths.sort(key=lambda p: (p.stem.replace("_fake", "").replace("_B", ""), p.name))
    return paths


def night_to_day_metrics(results_dir: Path, model_name: str, epoch: str, testB_dir: Path) -> tuple[float, float]:
    """Average MSE and SSIM over 220 night->day (fake vs real day in testB)."""
    img_dir = results_dir / model_name / f"test_{epoch}" / "images"
    if not img_dir.exists():
        return float("nan"), float("nan")
    fakes = collect_fake_paths(img_dir)
    reals = sorted(testB_dir.glob("*"))
    reals = [p for p in reals if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    reals = reals[: len(fakes)]
    if len(fakes) != len(reals) or len(fakes) == 0:
        return float("nan"), float("nan")
    mses, ssims = [], []
    for f, r in zip(fakes, reals):
        real = load_rgb(r)
        fake = load_rgb(f)
        mses.append(mse_channelwise_avg(real, fake))
        ssims.append(ssim_one(real, fake))
    return float(np.mean(mses)), float(np.mean(ssims))


def identity_fidelity(results_dir: Path, model_name_idt: str, epoch: str, testB_dir: Path) -> float:
    """Identity Fidelity = 1 - mean L1 over 220 day->day (fake = G_A(day), real = day)."""
    img_dir = results_dir / model_name_idt / f"test_{epoch}" / "images"
    if not img_dir.exists():
        return float("nan")
    fakes = collect_fake_paths(img_dir)
    reals = sorted(testB_dir.glob("*"))
    reals = [p for p in reals if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    reals = reals[: len(fakes)]
    if len(fakes) != len(reals) or len(fakes) == 0:
        return float("nan")
    l1s = []
    for f, r in zip(fakes, reals):
        real = load_rgb(r)
        fake = load_rgb(f)
        l1s.append(l1_one(real, fake))
    return 1.0 - float(np.mean(l1s))


def main():
    ap = argparse.ArgumentParser(description="Batch metric comparison for three models (MSE, SSIM, Identity).")
    ap.add_argument("--results_dir", type=Path, default=RESULTS, help="Results root")
    ap.add_argument("--dataroot", type=Path, default=DATAROOT, help="large_test dataset root (testA, testB)")
    ap.add_argument("--epochs", nargs=3, default=None, help="Epochs for mse, ssim, idt (default: 55 70 140)")
    args = ap.parse_args()

    testB = args.dataroot / "testB"
    if not testB.is_dir():
        print("Warning: testB not found at", testB, "- identity and night-to-day metrics may be missing.")

    epochs = args.epochs if args.epochs else [MODELS[i][1] for i in range(3)]
    rows = []
    for (model_name, epoch), ep in zip(MODELS, epochs):
        mse, ssim = night_to_day_metrics(args.results_dir, model_name, ep, testB)
        idt_fid = identity_fidelity(args.results_dir, f"{model_name}_idt", ep, testB)
        rows.append((model_name, mse, ssim, idt_fid))

    # Leaderboard table
    print("\n" + "=" * 70)
    print("  Performance Leaderboard (N=220 night-to-day, N=220 day-to-day identity)")
    print("=" * 70)
    print(f"  {'Model':<22}  {'Avg MSE (↓)':<14}  {'Avg SSIM (↑)':<14}  {'Identity Fidelity (↑)':<22}")
    print("-" * 70)
    for name, mse, ssim, idt in rows:
        mse_s = f"{mse:.4f}" if not np.isnan(mse) else "N/A"
        ssim_s = f"{ssim:.4f}" if not np.isnan(ssim) else "N/A"
        idt_s = f"{idt:.4f}" if not np.isnan(idt) else "N/A"
        print(f"  {name:<22}  {mse_s:<14}  {ssim_s:<14}  {idt_s:<22}")
    print("=" * 70)
    print("  MSE: lower is better.  SSIM & Identity Fidelity: higher is better.")
    print("  Identity Fidelity = 1 - L1(fake_day, real_day) for day->day pass.")
    print()

    # Optional: write CSV for slides
    out_csv = ROOT / "important_metrics" / "batch_leaderboard.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        f.write("model,avg_mse,avg_ssim,identity_fidelity\n")
        for name, mse, ssim, idt in rows:
            f.write(f"{name},{mse:.6f},{ssim:.6f},{idt:.6f}\n")
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
