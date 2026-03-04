"""
Copy night-time training images (including at least one Dark Zurich) to a folder and zip for testing.
Originals in the data folder are left unchanged.
"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT.parent / "test_night_samples.zip"
STAGING_DIR = ROOT.parent / "_test_night_staging"

TOTAL_WANTED = 4  # 3 other + 1 Dark Zurich (or 4 if we have zurich + 3 others)


def _dark_zurich_night_images():
    """Yield night image paths from raw/dark_zurich (train_night, test_night)."""
    for sub in ("train_night", "test_night"):
        d = ROOT / "raw" / "dark_zurich" / sub
        if d.exists():
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                yield p


def _other_night_images():
    """Yield night images from professor pair and processed (no zurich)."""
    prof_night = ROOT / "raw" / "professor_pair" / "night.jpg"
    if prof_night.exists():
        yield prof_night
    for sub in ("trainA", "testA"):
        d = ROOT / "processed" / sub
        if d.exists():
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                yield p
    night2day = ROOT.parent.parent / "datasets" / "night2day"
    if night2day.exists():
        for sub in ("trainA", "testA"):
            d = night2day / sub
            if d.exists():
                for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                    yield p


def collect_night_images(include_zurich: bool = True, max_count: int = TOTAL_WANTED):
    """Gather night images: at least one Dark Zurich if available, then others up to max_count."""
    zurich_list = [p for p in _dark_zurich_night_images() if p.exists()][:1]
    other_list = []
    seen = set()
    for p in _other_night_images():
        if not p.exists():
            continue
        key = (p.stat().st_size, p.name)
        if key in seen:
            continue
        seen.add(key)
        other_list.append(p)
        if len(other_list) >= max_count - len(zurich_list):
            break

    # Prefer: 1 Dark Zurich + (max_count - 1) others
    if include_zurich and zurich_list:
        return zurich_list + other_list[: max_count - 1]
    return other_list[:max_count]


def main():
    night_images = collect_night_images(include_zurich=True, max_count=TOTAL_WANTED)
    min_ok = 3
    if len(night_images) < min_ok:
        print(f"Found only {len(night_images)} night image(s). Need at least {min_ok} for the zip.")
        print("Places checked: data/raw/dark_zurich/train_night|test_night, professor_pair, processed, datasets/night2day")
        if night_images:
            print("Creating zip with available images.")
        else:
            print("No night images found. Add images and run again.")
            return

    # Detect Dark Zurich by path (raw/dark_zurich/...)
    zurich_prefix = (ROOT / "raw" / "dark_zurich").resolve()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, src in enumerate(night_images, 1):
            ext = src.suffix
            is_zurich = str(src.resolve()).startswith(str(zurich_prefix))
            base = f"night_sample_{i}_dark_zurich" if is_zurich else f"night_sample_{i}"
            dest = STAGING_DIR / f"{base}{ext}"
            shutil.copy2(src, dest)
            label = " (Dark Zurich)" if is_zurich else ""
            print(f"  Copied {src.name} -> {dest.name}{label}")

        print(f"Creating {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in STAGING_DIR.iterdir():
                if f.is_file():
                    zf.write(f, f.name)
        print(f"Done. Test images (copies) are in {ZIP_PATH}. Originals unchanged in data folder.")
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
