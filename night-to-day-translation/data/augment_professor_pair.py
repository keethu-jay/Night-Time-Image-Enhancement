"""
Augment a single night/day image pair into 400 synchronized pairs.
All transforms are applied identically to both images to preserve pairing
(pix2pix / paired image-to-image standard).
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Paths (edit if your layout differs)
#
# Output layout (pix2pix-style):
#   processed/
#     trainA/   <- night images for TRAINING
#     trainB/   <- day images for TRAINING (paired by filename)
#     testA/    <- night images for TESTING
#     testB/    <- day images for TESTING (paired by filename)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "professor_pair"
PROCESSED_DIR = ROOT / "processed"
TRAIN_A = PROCESSED_DIR / "trainA"   # night (training)
TRAIN_B = PROCESSED_DIR / "trainB"   # day (training)
TEST_A = PROCESSED_DIR / "testA"     # night (testing)
TEST_B = PROCESSED_DIR / "testB"     # day (testing)

NIGHT_NAME = "night.jpg"
DAY_NAME = "day.jpg"
NUM_PAIRS = 400
PROF_TRAIN = 220   # Professor pairs for training (for combined split: 900 Zurich + 220 Prof = 1,120)
PROF_TEST = 180    # Professor pairs for testing (for combined split: 100 Zurich + 180 Prof = 280)

# -----------------------------------------------------------------------------
# Synchronized augmentation (same transform on both images)
# -----------------------------------------------------------------------------


def load_pair():
    """Load night and day images; ensure same size."""
    night_path = RAW_DIR / NIGHT_NAME
    day_path = RAW_DIR / DAY_NAME
    if not night_path.exists() or not day_path.exists():
        raise FileNotFoundError(
            f"Expected {night_path} and {day_path}. Add night.jpg and day.jpg to data/raw/professor_pair/"
        )
    night = np.array(Image.open(night_path).convert("RGB"))
    day = np.array(Image.open(day_path).convert("RGB"))
    if night.shape != day.shape:
        # Resize day to match night
        day_im = Image.fromarray(day)
        day_im = day_im.resize((night.shape[1], night.shape[0]), Image.Resampling.LANCZOS)
        day = np.array(day_im)
    return night, day


def random_crop(night: np.ndarray, day: np.ndarray, crop_size: int):
    """Random crop the same region from both images."""
    h, w = night.shape[:2]
    if crop_size >= min(h, w):
        return night, day
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        night[top : top + crop_size, left : left + crop_size],
        day[top : top + crop_size, left : left + crop_size],
    )


def random_flip_h(night: np.ndarray, day: np.ndarray):
    """Random horizontal flip (same for both)."""
    if random.random() < 0.5:
        night = np.fliplr(night).copy()
        day = np.fliplr(day).copy()
    return night, day


def random_flip_v(night: np.ndarray, day: np.ndarray):
    """Random vertical flip (same for both)."""
    if random.random() < 0.5:
        night = np.flipud(night).copy()
        day = np.flipud(day).copy()
    return night, day


def random_rotate(night: np.ndarray, day: np.ndarray, max_angle_deg: float = 15):
    """Random rotation by the same angle (same for both)."""
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    if abs(angle) < 0.5:
        return night, day
    pil_night = Image.fromarray(night)
    pil_day = Image.fromarray(day)
    pil_night = pil_night.rotate(angle, Image.Resampling.BICUBIC, expand=False)
    pil_day = pil_day.rotate(angle, Image.Resampling.BICUBIC, expand=False)
    return np.array(pil_night), np.array(pil_day)


def random_scale_crop(night: np.ndarray, day: np.ndarray, scale_range=(0.85, 1.0), target_size=None):
    """
    Random scale then center-crop (or crop to target). Same scale/crop for both.
    Gives scale variation while keeping fixed output size for training.
    """
    h, w = night.shape[:2]
    target_size = target_size or min(h, w)
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < target_size or new_w < target_size:
        new_h, new_w = max(target_size, new_h), max(target_size, new_w)
    pil_night = Image.fromarray(night).resize((new_w, new_h), Image.Resampling.LANCZOS)
    pil_day = Image.fromarray(day).resize((new_w, new_h), Image.Resampling.LANCZOS)
    night_s, day_s = np.array(pil_night), np.array(pil_day)
    # Center crop to target_size
    nh, nw = night_s.shape[:2]
    top = (nh - target_size) // 2
    left = (nw - target_size) // 2
    night_s = night_s[top : top + target_size, left : left + target_size]
    day_s = day_s[top : top + target_size, left : left + target_size]
    return night_s, day_s


def apply_paired_augmentation(night: np.ndarray, day: np.ndarray, output_size: int):
    """
    One random augmented pair. All ops are applied identically to both images.
    """
    # Optional: random scale then center-crop for size diversity
    night, day = random_scale_crop(night, day, scale_range=(0.8, 1.0), target_size=output_size)
    night, day = random_flip_h(night, day)
    night, day = random_flip_v(night, day)
    night, day = random_rotate(night, day, max_angle_deg=12)
    # Final resize to exact output_size if rotation changed dimensions
    h, w = night.shape[:2]
    if h != output_size or w != output_size:
        night = np.array(Image.fromarray(night).resize((output_size, output_size), Image.Resampling.LANCZOS))
        day = np.array(Image.fromarray(day).resize((output_size, output_size), Image.Resampling.LANCZOS))
    return night, day


def main():
    random.seed(42)
    np.random.seed(42)

    num_train = PROF_TRAIN
    num_test = PROF_TEST

    TRAIN_A.mkdir(parents=True, exist_ok=True)
    TRAIN_B.mkdir(parents=True, exist_ok=True)
    TEST_A.mkdir(parents=True, exist_ok=True)
    TEST_B.mkdir(parents=True, exist_ok=True)

    night, day = load_pair()
    min_side = min(night.shape[0], night.shape[1])
    output_size = min(256, min_side)

    print(
        f"Loaded pair: night {night.shape}, day {day.shape}. Generating {NUM_PAIRS} pairs at {output_size}x{output_size}..."
    )
    print(f"Split: {num_train} train, {num_test} test\n")

    for i in range(NUM_PAIRS):
        n_aug, d_aug = apply_paired_augmentation(night.copy(), day.copy(), output_size)
        if i < num_train:
            base_name = f"professor_pair_{i:04d}"
            Image.fromarray(n_aug).save(TRAIN_A / f"{base_name}_night.png")
            Image.fromarray(d_aug).save(TRAIN_B / f"{base_name}_day.png")
        else:
            j = i - num_train
            base_name = f"professor_pair_{j:04d}"
            Image.fromarray(n_aug).save(TEST_A / f"{base_name}_night.png")
            Image.fromarray(d_aug).save(TEST_B / f"{base_name}_day.png")

    print("Done. Pairs saved to:")
    print(f"  TRAIN:  {TRAIN_A}  <->  {TRAIN_B}  ({num_train} pairs)")
    print(f"  TEST:   {TEST_A}  <->  {TEST_B}  ({num_test} pairs)")


if __name__ == "__main__":
    main()
