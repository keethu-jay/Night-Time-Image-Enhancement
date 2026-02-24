"""
Resize all images in datasets/night2day/ to 256x256 if larger (GPU training),
then zip the folder.
"""

from pathlib import Path
import zipfile
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parent
NIGHT2DAY = ROOT.parent.parent / "datasets" / "night2day"
MAX_SIZE = 256


def resize_dir(d: Path) -> int:
    resized = 0
    for p in list(d.glob("*.png")) + list(d.glob("*.jpg")):
        with Image.open(p) as im:
            w, h = im.size
            if w > MAX_SIZE or h > MAX_SIZE:
                im = im.convert("RGB")
                im = im.resize((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
                im.save(p)
                resized += 1
    return resized


def main():
    if not NIGHT2DAY.exists():
        print(f"Not found: {NIGHT2DAY}. Run organize_night2day.py first.")
        return

    total = 0
    for sub in ("trainA", "trainB", "testA", "testB"):
        d = NIGHT2DAY / sub
        if d.exists():
            total += resize_dir(d)
    if total:
        print(f"Resized {total} images to {MAX_SIZE}x{MAX_SIZE}")
    else:
        print(f"All images already <= {MAX_SIZE}x{MAX_SIZE}")

    zip_path = NIGHT2DAY.parent / "night2day_data.zip"
    print(f"Zipping {NIGHT2DAY} -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in NIGHT2DAY.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(NIGHT2DAY.parent))
    print(f"Done. Created {zip_path}")


if __name__ == "__main__":
    main()
