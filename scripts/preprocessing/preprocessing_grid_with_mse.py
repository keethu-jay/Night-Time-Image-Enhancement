#!/usr/bin/env python3
"""
Builds the seven-panel comparison image for traditional enhancement methods. Each
panel shows one variant and its MSE vs the paired day image. MSE is computed on
luminance only (so grayscale outputs like retinex are compared fairly). Low MSE
means the brightness pattern is similar; it does not guarantee the image looks
like the day (color and alignment matter). Run from project root:
python scripts/preprocessing/preprocessing_grid_with_mse.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SIZE = (256, 256)


def load_rgb(path: Path) -> np.ndarray:
    """Load image from path, resize to 256x256, return RGB in [0, 1] as float."""
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64) / 255.0


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to luminance (single channel). Standard Rec. 601 weights."""
    return np.clip(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 1
    ).astype(np.float64)


def mse_luminance(a: np.ndarray, b: np.ndarray) -> float:
    """
    MSE between luminance of a and b. Use this so grayscale outputs (e.g. retinex)
    are compared fairly to the day image; otherwise gray-vs-RGB can give
    misleadingly low MSE. Also requires night and day to be the same scene/crop.
    """
    if a.shape != b.shape:
        b = cv2.resize(
            (np.clip(b, 0, 1) * 255).astype(np.uint8),
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        b = b.astype(np.float64) / 255.0
    lum_a = rgb_to_luminance(a) if a.ndim == 3 else np.asarray(a, dtype=np.float64)
    lum_b = rgb_to_luminance(b) if b.ndim == 3 else np.asarray(b, dtype=np.float64)
    return float(np.mean((lum_a - lum_b) ** 2))


def to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Float [0,1] RGB -> uint8 RGB for OpenCV (BGR)."""
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def from_bgr(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float64) / 255.0


def enhance_hsv(night_rgb: np.ndarray) -> np.ndarray:
    """HSV visualization: convert to HSV, scale V for visibility, back to RGB."""
    u8 = to_uint8_rgb(night_rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    v = np.clip(v * 1.5, 0, 255)
    s = np.clip(s * 1.2, 0, 255)
    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return from_bgr(bgr)


def enhance_denoised(night_rgb: np.ndarray) -> np.ndarray:
    """Denoise with fastNlMeansDenoisingColored."""
    u8 = to_uint8_rgb(night_rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    try:
        out = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
    except (TypeError, cv2.error):
        out = cv2.bilateralFilter(bgr, 9, 75, 75)
    return from_bgr(out)


def enhance_gamma(night_rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Gamma correction: I_out = I_in^gamma (gamma>1 brightens)."""
    out = np.power(np.clip(night_rgb, 1e-6, 1), gamma)
    return np.clip(out, 0, 1).astype(np.float64)


def enhance_clahe(night_rgb: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """CLAHE on L channel in LAB, then merge back."""
    u8 = to_uint8_rgb(night_rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return from_bgr(bgr)


def _single_scale_retinex(rgb: np.ndarray, sigma: int) -> np.ndarray:
    """Single-scale retinex: log(I) - log(L) with Gaussian blur L."""
    u8 = to_uint8_rgb(rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) + 1
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    blur = np.maximum(blur, 1)
    log_r = np.log10(gray) - np.log10(blur)
    log_r = (log_r - log_r.min()) / (log_r.max() - log_r.min() + 1e-8)
    out_bgr = cv2.cvtColor((log_r * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return from_bgr(out_bgr)


def enhance_single_scale(night_rgb: np.ndarray, sigma: int = 30) -> np.ndarray:
    """Single-scale retinex (one sigma)."""
    return _single_scale_retinex(night_rgb, sigma)


def enhance_multi_scale(night_rgb: np.ndarray, sigmas: tuple[int, ...] = (15, 80, 250)) -> np.ndarray:
    """Multi-scale retinex: average of single-scale at several sigmas, then normalize."""
    u8 = to_uint8_rgb(night_rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) + 1
    acc = np.zeros_like(gray)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(gray, (0, 0), sigma)
        blur = np.maximum(blur, 1)
        acc += np.log10(gray) - np.log10(blur)
    acc /= len(sigmas)
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    out_bgr = cv2.cvtColor((np.clip(acc, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return from_bgr(out_bgr)


def main():
    ap = argparse.ArgumentParser(description="7-panel preprocessing grid with MSE vs day image.")
    ap.add_argument("--night", type=Path, default=None, help="Night image path.")
    ap.add_argument("--day", type=Path, default=None, help="Day (ground truth) image for MSE.")
    ap.add_argument("--out", type=Path, default=None, help="Output image path.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    metrics_dir = root / "important_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prof_night = root / "night-to-day-translation" / "data" / "raw" / "professor_pair" / "night.jpg"
    prof_day = root / "night-to-day-translation" / "data" / "raw" / "professor_pair" / "day.jpg"

    if args.night and args.night.exists():
        night_path = args.night
    elif prof_night.exists():
        night_path = prof_night
    else:
        test_a = root / "datasets" / "night2day" / "testA"
        test_b = root / "datasets" / "night2day" / "testB"
        if test_a.exists() and test_b.exists():
            files = sorted(test_a.glob("*.png")) or sorted(test_a.glob("*.jpg"))
            first = next(iter(files), None) if files else None
            if first:
                night_path = first
            else:
                raise SystemExit("No night image found. Put professor pair in night-to-day-translation/data/raw/professor_pair/ or pass --night.")
        else:
            raise SystemExit("No night image found. Put professor pair in night-to-day-translation/data/raw/professor_pair/ or pass --night.")
        if not night_path.exists():
            raise SystemExit("No night image found. Pass --night path/to/night.png")

    if args.day and args.day.exists():
        day_path = args.day
    elif night_path == prof_night and prof_day.exists():
        day_path = prof_day
    else:
        day_path = None
        test_b = root / "datasets" / "night2day" / "testB"
        if test_b.exists() and night_path.parent.name == "testA":
            day_path = test_b / night_path.name
        if not day_path or not day_path.exists():
            day_path = prof_day if prof_day.exists() else None
    night_rgb = load_rgb(night_path)
    if day_path and day_path.exists():
        day_rgb = load_rgb(day_path)
        has_gt = True
    else:
        day_rgb = None
        has_gt = False

    panels = [
        ("Night Image", night_rgb),
        ("HSV Image", enhance_hsv(night_rgb)),
        ("Denoised Image", enhance_denoised(night_rgb)),
        ("Gamma Correction", enhance_gamma(night_rgb)),
        ("CLAHE Image", enhance_clahe(night_rgb)),
        ("Single Scale Image", enhance_single_scale(night_rgb)),
        ("Multi Scale Image", enhance_multi_scale(night_rgb)),
    ]

    if has_gt:
        mses = [mse_luminance(im, day_rgb) for _, im in panels]
    else:
        mses = [None] * len(panels)

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle(
        "MSE = luminance vs day (same scene). Low MSE = similar brightness, not necessarily similar look.",
        fontsize=9,
        y=1.02,
    )
    axes = axes.flatten()
    for i, (title, im) in enumerate(panels):
        ax = axes[i]
        ax.imshow(np.clip(im, 0, 1))
        mse_str = (
            f"MSE (lum) = {mses[i]:.4f}"
            if mses[i] is not None
            else "MSE = N/A (no GT)"
        )
        ax.set_title(f"{title}\n{mse_str}", fontsize=10)
        ax.axis("off")
    for j in range(len(panels), 9):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = args.out or (metrics_dir / "preprocessing_grid_with_mse.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)
    if has_gt:
        print(
            "MSE (luminance vs day): Night = {:.4f}, HSV = {:.4f}, Denoised = {:.4f}, "
            "Gamma = {:.4f}, CLAHE = {:.4f}, Single = {:.4f}, Multi = {:.4f}".format(
                *mses
            )
        )
    else:
        print("No day image provided; MSE labels set to N/A. Pass --day for MSE vs ground truth.")


if __name__ == "__main__":
    main()
