#!/usr/bin/env python3
"""
Grading script for professor/TA: run your own night (and optional day) image through
the three best models (best MSE, best SSIM, best identity) and get MSE/SSIM results.

Use your own images so you can grade submissions or try different night/day pairs.
We run inference with the pretrained best-MSE, best-SSIM, and best-identity weights,
then compare each fake day output to your ground-truth day image if you provide one.

Run from project root:
  python scripts/metrics/run_grading.py --night path/to/your_night.jpg [--day path/to/your_day.jpg]
  python scripts/metrics/run_grading.py --night my_night.png --day my_day.png --out-dir grading_results

If you omit --day we still run all three models and save the fake-day images; we just
cannot compute MSE/SSIM without a ground truth. Outputs go to --out-dir (default:
important_metrics/grading/) and include a summary table plus one verification image
per model.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

# Same best epochs as run_batch_inference and build_important_metrics
BEST_MSE_EPOCH = "55"
BEST_SSIM_EPOCH = "70"
BEST_IDT_EPOCH = "140"
SOURCE_ROUND = "round2"
MODEL_CONFIGS = [
    ("model_best_mse", BEST_MSE_EPOCH, "Best MSE"),
    ("model_best_ssim", BEST_SSIM_EPOCH, "Best SSIM"),
    ("model_best_idt", BEST_IDT_EPOCH, "Best Identity"),
]

SIZE = (256, 256)
GRADING_INPUT = ROOT / "datasets" / "grading_input" / "testA"


def load_rgb(path: Path) -> np.ndarray:
    """Load image, resize to SIZE, return RGB float in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64) / 255.0


def mse_channelwise(real_rgb: np.ndarray, fake_rgb: np.ndarray) -> tuple[float, float, float, float]:
    """Channel-wise MSE (R, G, B, average). Used for grading report."""
    mse_r = float(np.mean((real_rgb[..., 0] - fake_rgb[..., 0]) ** 2))
    mse_g = float(np.mean((real_rgb[..., 1] - fake_rgb[..., 1]) ** 2))
    mse_b = float(np.mean((real_rgb[..., 2] - fake_rgb[..., 2]) ** 2))
    mse_avg = (mse_r + mse_g + mse_b) / 3.0
    return (mse_r, mse_g, mse_b, mse_avg)


def ensure_checkpoint(name: str, epoch: str) -> None:
    """Copy round2 epoch weights into checkpoints/name if not already there."""
    dest_dir = CHECKPOINTS / name
    dest_pth = dest_dir / f"{epoch}_net_G_A.pth"
    if dest_pth.exists():
        return
    src = CHECKPOINTS / SOURCE_ROUND / f"{epoch}_net_G_A.pth"
    if not src.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {src}. Run training or copy round2 weights into checkpoints/round2/."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_pth)
    print("Prepared checkpoint:", dest_pth)


def run_inference(name: str, epoch: str, dataroot: Path, dry_run: bool) -> Path | None:
    """Run test.py for one model; return path to the first fake image, or None if dry_run."""
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot", str(dataroot),
        "--name", name,
        "--checkpoints_dir", str(CHECKPOINTS),
        "--results_dir", str(RESULTS),
        "--model", "test",
        "--model_suffix", "_A",
        "--no_dropout",
        "--epoch", epoch,
        "--num_test", "1",
    ]
    print("Run:", " ".join(cmd))
    if dry_run:
        return None
    r = subprocess.run(cmd, cwd=SRC, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"test.py exited with code {r.returncode}")

    # Output is under results/name/test_<epoch>/images/; first image is *fake_B.png or *_fake.png
    images_dir = RESULTS / name / f"test_{epoch}" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Expected results at {images_dir}")
    for f in images_dir.glob("*_fake_B.png"):
        return f
    for f in images_dir.glob("*_fake*.png"):
        return f
    raise FileNotFoundError(f"No fake image found in {images_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Run grading: your night (and optional day) image through best MSE, SSIM, and Identity models."
    )
    ap.add_argument(
        "--night",
        type=Path,
        required=True,
        help="Path to your night image (required).",
    )
    ap.add_argument(
        "--day",
        type=Path,
        default=None,
        help="Path to ground-truth day image. If provided, we compute MSE and SSIM for each model.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for outputs (table, verification images). Default: important_metrics/grading/",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print inference commands; do not run or compute metrics.",
    )
    args = ap.parse_args()

    night_path = args.night.resolve()
    if not night_path.exists():
        raise SystemExit(f"Night image not found: {night_path}")

    day_path = args.day.resolve() if args.day else None
    if args.day and not day_path.exists():
        raise SystemExit(f"Day image not found: {day_path}")

    out_dir = args.out_dir or (ROOT / "important_metrics" / "grading")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_dir)

    # Copy night image into a fixed testA folder so test.py can load it
    GRADING_INPUT.mkdir(parents=True, exist_ok=True)
    # Use a fixed name so we can find the fake output
    staging_night = GRADING_INPUT / "grading_night.png"
    shutil.copy2(night_path, staging_night)
    print("Staged night image at", staging_night)
    dataroot = GRADING_INPUT.parent

    # Ensure all three checkpoints exist
    for name, epoch, _ in MODEL_CONFIGS:
        ensure_checkpoint(name, epoch)

    # Run inference for each model and collect fake image paths
    fake_paths = []
    for name, epoch, label in MODEL_CONFIGS:
        print("\n---", label, "(", name, "epoch", epoch, ") ---")
        try:
            fp = run_inference(name, epoch, dataroot, args.dry_run)
            fake_paths.append((label, name, epoch, fp))
        except Exception as e:
            print("Error:", e)
            fake_paths.append((label, name, epoch, None))

    if args.dry_run:
        print("\nDry run done. No metrics computed.")
        return

    # Load ground truth once if provided
    day_rgb = load_rgb(day_path) if day_path else None

    # Compute metrics per model and write table
    rows = []
    for label, name, epoch, fake_path in fake_paths:
        if fake_path is None or not fake_path.exists():
            rows.append((label, name, epoch, None, None, None, None))
            continue
        fake_rgb = load_rgb(fake_path)
        # Save a copy into out_dir for the professor
        out_fake = out_dir / f"grading_{name}_epoch{epoch}_fake.png"
        shutil.copy2(fake_path, out_fake)
        if day_rgb is not None:
            mse_r, mse_g, mse_b, mse_avg = mse_channelwise(day_rgb, fake_rgb)
            if ssim is not None:
                ssim_val = ssim(day_rgb, fake_rgb, channel_axis=2, data_range=1.0)
            else:
                ssim_val = 0.0
            similarity_pct = 100.0 * (1.0 - mse_avg)
            fidelity_pct = ssim_val * 100.0
            rows.append((label, name, epoch, mse_avg, ssim_val, similarity_pct, fidelity_pct))
        else:
            rows.append((label, name, epoch, None, None, None, None))

    # Print table
    print("\n" + "=" * 70)
    print("GRADING RESULTS (your night image -> fake day, vs your day image)")
    print("=" * 70)
    if day_rgb is not None:
        print(f"{'Model':<18} {'MSE (avg)':<12} {'SSIM':<10} {'Similarity %':<14} {'Fidelity (SSIM*100)':<22}")
        print("-" * 70)
        for label, name, epoch, mse_avg, ssim_val, sim_pct, fid_pct in rows:
            if mse_avg is not None:
                print(f"{label:<18} {mse_avg:<12.6f} {ssim_val:<10.4f} {sim_pct:<14.2f} {fid_pct:<22.2f}")
            else:
                print(f"{label:<18} (no fake output)")
    else:
        print("No --day image provided; only fake-day images were saved. Re-run with --day to get MSE/SSIM.")
        for label, name, epoch, *_ in rows:
            print("  ", label, "->", out_dir / f"grading_{name}_epoch{epoch}_fake.png")
    print("=" * 70)
    print("Fake-day images saved under:", out_dir)

    # Write a short text report
    report_path = out_dir / "grading_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Grading run: night = %s\n" % night_path)
        if day_path:
            f.write("              day  = %s\n" % day_path)
        f.write("\n")
        if day_rgb is not None:
            f.write("%-18s %-12s %-10s %-14s %-22s\n" % (
                "Model", "MSE (avg)", "SSIM", "Similarity %", "Fidelity (SSIM*100)"))
            f.write("-" * 70 + "\n")
            for label, name, epoch, mse_avg, ssim_val, sim_pct, fid_pct in rows:
                if mse_avg is not None:
                    f.write("%-18s %-12.6f %-10.4f %-14.2f %-22.2f\n" % (
                        label, mse_avg, ssim_val, sim_pct, fid_pct))
        f.write("\nFake images: grading_<model>_epoch<epoch>_fake.png\n")
    print("Report written to", report_path)


if __name__ == "__main__":
    main()
