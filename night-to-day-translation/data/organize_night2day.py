"""
Organize combined training data into datasets/night2day/ with the specified split.
Resizes all images to 256x256 if larger (crucial for GPU training).

  datasets/night2day/
  ├── trainA/  # 1,120 total (900 Zurich Night + 220 Prof Night)
  ├── trainB/  # 1,120 total (900 Zurich Day + 220 Prof Day)
  ├── testA/   # 280 total (100 Zurich Night + 180 Prof Night)
  └── testB/   # 280 total (100 Zurich Day + 180 Prof Day)

Pairs use matching filenames (pix2pix convention): trainA/x.png <-> trainB/x.png

Expects:
- Zurich images in ZURICH_DIR with subdirs: train_night, train_day, test_night, test_day
  (900 train pairs, 100 test pairs). Pairs matched by sorted order (same index = same scene).
- Professor augmented pairs in PROF_DIR (from augment_professor_pair.py)
  (220 train, 180 test)
"""

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent.parent / "datasets" / "night2day"  # project root / datasets / night2day

# Source directories (edit if your layout differs)
ZURICH_DIR = ROOT / "raw" / "dark_zurich"   # Zurich split: train_night, train_day, test_night, test_day
PROF_DIR = ROOT / "processed"               # Output from augment_professor_pair.py

# Expected counts
ZURICH_TRAIN = 900
ZURICH_TEST = 100
PROF_TRAIN = 220
PROF_TEST = 180
MAX_SIZE = 256  # Resize images larger than this (GPU training)


def resize_if_needed(img_dir: Path, max_size: int = MAX_SIZE):
    """Resize images larger than max_size to max_size x max_size (in-place)."""
    resized = 0
    for p in list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")):
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                if w > max_size or h > max_size:
                    im = im.convert("RGB")
                    im = im.resize((max_size, max_size), Image.Resampling.LANCZOS)
                    im.save(p)
                    resized += 1
        except (OSError, ValueError):
            pass  # Skip corrupt images
    return resized


def copy_paired(night_dir: Path, day_dir: Path, dstA: Path, dstB: Path, prefix: str, limit: int):
    """Copy paired night/day images with matching filenames for pix2pix.
    Pairs by matching filename when possible; otherwise by sorted index."""
    dstA.mkdir(parents=True, exist_ok=True)
    dstB.mkdir(parents=True, exist_ok=True)
    night_list = sorted(night_dir.glob("*.png")) + sorted(night_dir.glob("*.jpg"))
    day_list = sorted(day_dir.glob("*.png")) + sorted(day_dir.glob("*.jpg"))
    night_by_name = {p.name: p for p in night_list}
    day_by_name = {p.name: p for p in day_list}
    common = sorted(set(night_by_name) & set(day_by_name))[:limit]
    if common:
        pairs = [(night_by_name[n], day_by_name[n]) for n in common]
    else:
        pairs = list(zip(night_list[:limit], day_list[:limit]))
    for i, (npath, dpath) in enumerate(pairs):
        base = f"{prefix}{i:05d}.png"
        shutil.copy2(npath, dstA / base)
        shutil.copy2(dpath, dstB / base)
    return len(pairs)


def main():
    trainA = OUTPUT_DIR / "trainA"
    trainB = OUTPUT_DIR / "trainB"
    testA = OUTPUT_DIR / "testA"
    testB = OUTPUT_DIR / "testB"

    # Clear output dirs for clean split (Zurich first, then Prof)
    for d in (trainA, trainB, testA, testB):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # --- Zurich first: 900 train, 100 test ---
    zurich_train_night = ZURICH_DIR / "train_night"
    zurich_train_day = ZURICH_DIR / "train_day"
    zurich_test_night = ZURICH_DIR / "test_night"
    zurich_test_day = ZURICH_DIR / "test_day"

    if zurich_train_night.exists() and zurich_train_day.exists():
        n = copy_paired(zurich_train_night, zurich_train_day, trainA, trainB, "zurich_", ZURICH_TRAIN)
        print(f"Copied {n} Zurich train pairs -> trainA/trainB")
    if zurich_test_night.exists() and zurich_test_day.exists():
        n = copy_paired(zurich_test_night, zurich_test_day, testA, testB, "zurich_", ZURICH_TEST)
        print(f"Copied {n} Zurich test pairs -> testA/testB")

    # --- Professor second: 220 train, 180 test ---
    prof_trainA = PROF_DIR / "trainA"
    prof_trainB = PROF_DIR / "trainB"
    prof_testA = PROF_DIR / "testA"
    prof_testB = PROF_DIR / "testB"

    def copy_prof_pairs(night_dir, day_dir, dstA, dstB, limit):
        night_imgs = sorted(night_dir.glob("*_night.png"))[:limit]
        for p in night_imgs:
            idx = p.stem.replace("professor_pair_", "").replace("_night", "")
            shutil.copy2(p, dstA / f"prof_{idx}.png")
        day_imgs = sorted(day_dir.glob("*_day.png"))[:limit]
        for p in day_imgs:
            idx = p.stem.replace("professor_pair_", "").replace("_day", "")
            shutil.copy2(p, dstB / f"prof_{idx}.png")
        return len(night_imgs)

    if prof_trainA.exists() and prof_trainB.exists():
        n = copy_prof_pairs(prof_trainA, prof_trainB, trainA, trainB, PROF_TRAIN)
        print(f"Copied {n} Prof train pairs -> trainA/trainB")
    if prof_testA.exists() and prof_testB.exists():
        n = copy_prof_pairs(prof_testA, prof_testB, testA, testB, PROF_TEST)
        print(f"Copied {n} Prof test pairs -> testA/testB")

    # Resize all output images to 256x256 if larger (crucial for GPU training)
    total_resized = 0
    for d in (trainA, trainB, testA, testB):
        total_resized += resize_if_needed(d, MAX_SIZE)
    if total_resized:
        print(f"Resized {total_resized} images to {MAX_SIZE}x{MAX_SIZE}")

    # Summary
    trainA_count = len(list(trainA.glob("*.png"))) + len(list(trainA.glob("*.jpg")))
    trainB_count = len(list(trainB.glob("*.png"))) + len(list(trainB.glob("*.jpg")))
    testA_count = len(list(testA.glob("*.png"))) + len(list(testA.glob("*.jpg")))
    testB_count = len(list(testB.glob("*.png"))) + len(list(testB.glob("*.jpg")))

    print(f"\nDone. Output: {OUTPUT_DIR}")
    print(f"  trainA: {trainA_count} images (target: 1,120)")
    print(f"  trainB: {trainB_count} images (target: 1,120)")
    print(f"  testA:  {testA_count} images (target: 280)")
    print(f"  testB:  {testB_count} images (target: 280)")


if __name__ == "__main__":
    main()
