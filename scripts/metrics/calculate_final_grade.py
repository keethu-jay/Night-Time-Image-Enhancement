#!/usr/bin/env python3
"""
Compares the real day image (ground truth) to the fake day image (model output).
Computes channel-wise MSE and SSIM, prints Fidelity Score (SSIM * 100), and saves
a side-by-side visual: Night input | Real day | Fake day with MSE in the bar.
Run from project root: python scripts/metrics/calculate_final_grade.py
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


SIZE = (256, 256)


def load_grayscale(path: Path) -> np.ndarray:
    """Load image, resize to SIZE, convert to grayscale by averaging RGB; return float [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float64) / 255.0
    gray = np.mean(arr[..., :3], axis=-1)
    return gray


def load_rgb(path: Path) -> np.ndarray:
    """Load image, resize to SIZE, return RGB float [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64) / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel MSE."""
    return float(np.mean((a - b) ** 2))


def mse_channelwise(real_rgb: np.ndarray, fake_rgb: np.ndarray) -> tuple[float, float, float, float]:
    """
    Channel-wise MSE: compute MSE separately for R, G, B, then average.
    real_rgb, fake_rgb: shape (H, W, 3), float [0, 1].
    Returns (MSE_R, MSE_G, MSE_B, MSE_avg).
    """
    mse_r = float(np.mean((real_rgb[..., 0] - fake_rgb[..., 0]) ** 2))
    mse_g = float(np.mean((real_rgb[..., 1] - fake_rgb[..., 1]) ** 2))
    mse_b = float(np.mean((real_rgb[..., 2] - fake_rgb[..., 2]) ** 2))
    mse_avg = (mse_r + mse_g + mse_b) / 3.0
    return (mse_r, mse_g, mse_b, mse_avg)


def main():
    ap = argparse.ArgumentParser(
        description="Compute MSE and SSIM between real_day and fake_day; save visual verification."
    )
    ap.add_argument(
        "--real",
        default=None,
        help="Path to real (ground truth) day image. Default: professor pair day.jpg.",
    )
    ap.add_argument(
        "--fake",
        default=None,
        help="Path to fake (model) day image. Default: results/round2/test_latest/images/*_fake.png.",
    )
    ap.add_argument(
        "--night",
        default=None,
        help="Path to night input image for the visual. Default: professor pair night.jpg.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path for the visual verification image. Default: final_verification.png in project root.",
    )
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="Compute SSIM in RGB (default: grayscale). Often gives higher, more interpretable Fidelity for color images.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Print only MSE and SSIM on one line for scripting (e.g. MSE=0.076 SSIM=0.204).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    prof = root / "night-to-day-translation" / "data" / "raw" / "professor_pair"

    real_path = Path(args.real) if args.real else (prof / "day.jpg")
    night_path = Path(args.night) if args.night else (prof / "night.jpg")

    if args.fake:
        fake_path = Path(args.fake)
    else:
        # Default: use fake that is G(night)→day. Prefer night_fake.png (from night input); else most recent *_fake.png
        results_dir = root / "results"
        fake_path = None
        latest_mtime = 0
        night_fake_path = None
        night_fake_mtime = 0
        if results_dir.exists():
            for res_dir in results_dir.iterdir():
                if not res_dir.is_dir():
                    continue
                for sub in res_dir.iterdir():
                    if not sub.is_dir():
                        continue
                    images_dir = sub / "images"
                    if not images_dir.exists():
                        continue
                    for f in images_dir.glob("*_fake.png"):
                        mtime = f.stat().st_mtime
                        if "night" in f.stem.lower() and mtime > night_fake_mtime:
                            night_fake_mtime = mtime
                            night_fake_path = f
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            fake_path = f
        if night_fake_path and night_fake_path.exists():
            fake_path = night_fake_path
        if not fake_path or not fake_path.exists():
            raise SystemExit(
                "Fake day image not found under results/<any>/test_latest/images/*_fake.png\n"
                "Run inference first (see scripts/inference/RUN_INFERENCE.md), or pass --fake path/to/fake.png"
            )
        if not args.quiet:
            if night_fake_path and fake_path == night_fake_path:
                print("Using fake image (G(night)→day):", fake_path)
            else:
                print("Using fake image (latest by mtime):", fake_path)

    # Load RGB for channel-wise MSE (required for grading)
    real_rgb = load_rgb(real_path)
    fake_rgb = load_rgb(fake_path)
    mse_r, mse_g, mse_b, mse_avg = mse_channelwise(real_rgb, fake_rgb)

    if args.rgb:
        real_g = load_grayscale(real_path)
        fake_g = load_grayscale(fake_path)
        if ssim is None:
            ssim_val = 0.0
            print("Warning: skimage not installed; SSIM set to 0. Install with: pip install scikit-image")
        else:
            ssim_val = ssim(real_rgb, fake_rgb, channel_axis=2, data_range=1.0)
    else:
        real_g = load_grayscale(real_path)
        fake_g = load_grayscale(fake_path)
        if ssim is None:
            ssim_val = 0.0
            print("Warning: skimage not installed; SSIM set to 0. Install with: pip install scikit-image")
        else:
            ssim_val = ssim(real_g, fake_g, data_range=1.0)

    fidelity_pct = ssim_val * 100.0
    # Similarity % from channel-wise MSE (for "percent over 80%" grading: 100*(1 - MSE) on [0,1] images)
    similarity_pct = 100.0 * (1.0 - mse_avg)

    if args.quiet:
        print(f"MSE_R={mse_r:.6f} MSE_G={mse_g:.6f} MSE_B={mse_b:.6f} MSE_avg={mse_avg:.6f} SSIM={ssim_val:.6f}")
    else:
        print("Channel-wise MSE (report this):")
        print(f"  MSE_R = {mse_r:.6f},  MSE_G = {mse_g:.6f},  MSE_B = {mse_b:.6f}")
        print(f"  MSE (average) = (MSE_R + MSE_G + MSE_B) / 3 = {mse_avg:.6f}")
        print("Similarity % (1 - MSE)×100 (aim >80%):", f"{similarity_pct:.2f}%")
        print("SSIM:", ssim_val, "(RGB)" if args.rgb else "(averaged RGB)")
        print("Fidelity Score (SSIM × 100):", f"{fidelity_pct:.2f}%")

    if args.quiet:
        return

    # Visual: Night | Real Day | Fake Day, with MSE on top
    out_path = Path(args.out) if args.out else (root / "final_verification.png")
    night_img = Image.open(night_path).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
    real_img_pil = Image.open(real_path).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
    fake_img_pil = Image.open(fake_path).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)

    top_bar = 36
    label_bar = 24
    w, h = SIZE[0] * 3, top_bar + SIZE[1] + label_bar
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(night_img, (0, top_bar))
    canvas.paste(real_img_pil, (SIZE[0], top_bar))
    canvas.paste(fake_img_pil, (SIZE[0] * 2, top_bar))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.rectangle([0, 0, w, top_bar - 1], fill=(240, 240, 240))
        draw.text((10, 8), f"MSE(R,G,B avg) = {mse_avg:.4f}  |  Similarity % = {similarity_pct:.1f}%  |  SSIM = {ssim_val:.4f}", fill=(0, 0, 0), font=font)
        y_label = top_bar + SIZE[1] + 4
        draw.rectangle([0, top_bar + SIZE[1], w, h - 1], fill=(60, 60, 60))
        draw.text((SIZE[0] // 2 - 25, y_label), "Night", fill=(255, 255, 255), font=font)
        draw.text((SIZE[0] + SIZE[0] // 2 - 35, y_label), "Real Day", fill=(255, 255, 255), font=font)
        draw.text((SIZE[0] * 2 + SIZE[0] // 2 - 35, y_label), "Fake Day", fill=(255, 255, 255), font=font)
    except Exception:
        pass

    canvas.save(out_path)
    print("Saved visual verification:", out_path)


if __name__ == "__main__":
    main()
