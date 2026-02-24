"""
Prepare Finalized_Dark_Zurich for organize_night2day.

Place Finalized_Dark_Zurich in one of:
  - night-to-day-translation/data/raw/dark_zurich/Finalized_Dark_Zurich
  - Night-Time-Image-Enhancement/dark_zurich/Finalized_Dark_Zurich
  - Night-Time-Image-Enhancement/Finalized_Dark_Zurich
Or pass path: python prepare_dark_zurich.py --path "C:/path/to/Finalized_Dark_Zurich"

Reads from Finalized_Dark_Zurich structure:
  Finalized_Dark_Zurich/
    day/
      GOPR0345/
        GOPR0345_frame_0001.png, ...
    night/
      GOPR0345/
        GOPR0345_frame_0001.png, ...  (same filename = paired)

Outputs to raw/dark_zurich/:
  train_night/, train_day/ (900 pairs)
  test_night/, test_day/ (100 pairs)

Also resizes to 256x256 if larger (GPU training).
"""

import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
# Check: data/raw/Finalized_Dark_Zurich, data/raw/dark_zurich/Finalized_Dark_Zurich, etc.
FINALIZED_PATHS = [
    ROOT / "raw" / "Finalized_Dark_Zurich",  # where you have it
    ROOT / "raw" / "dark_zurich" / "Finalized_Dark_Zurich",
    ROOT.parent.parent / "dark_zurich" / "Finalized_Dark_Zurich",
    ROOT.parent.parent / "Finalized_Dark_Zurich",
]
OUTPUT_DIR = ROOT / "raw" / "dark_zurich"
ZURICH_TRAIN = 900
ZURICH_TEST = 100
MAX_SIZE = 256


def _frame_key(p: Path) -> str:
    """Extract frame id from filename, e.g. GOPR0345_frame_000150_rgb_anon.png -> 000150."""
    s = p.stem
    if "_frame_" in s:
        parts = s.split("_frame_")
        if len(parts) == 2:
            return parts[1].split("_")[0]  # e.g. 000150
    return p.name


def find_pairs(night_root: Path, day_root: Path):
    """Collect (night_path, day_path). Pairs by filename, or by (folder_index, frame_id)."""
    night_list = [p for p in night_root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    day_list = [p for p in day_root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    night_by_name = {p.name: p for p in night_list}
    day_by_name = {p.name: p for p in day_list}
    common = sorted(set(night_by_name) & set(day_by_name))
    if common:
        return [(night_by_name[n], day_by_name[n]) for n in common]

    # Pair by folder index + frame_id (e.g. GOPR0345 day <-> GOPR0351 night)
    def by_folder(p):
        return p.parent.name
    night_by_folder = {}
    for p in night_list:
        k = (by_folder(p), _frame_key(p))
        night_by_folder[k] = p
    day_by_folder = {}
    for p in day_list:
        k = (by_folder(p), _frame_key(p))
        day_by_folder[k] = p
    # Match by (folder_rank, frame_id) - sort folders and align by index
    n_folders = sorted({by_folder(p) for p in night_list})
    d_folders = sorted({by_folder(p) for p in day_list})
    folder_map = dict(zip(n_folders, d_folders)) if len(n_folders) == len(d_folders) else {}
    pairs = []
    for np in night_list:
        nf, fid = by_folder(np), _frame_key(np)
        df = folder_map.get(nf)
        if df:
            key = (df, fid)
            if key in day_by_folder:
                pairs.append((np, day_by_folder[key]))
    return sorted(pairs, key=lambda x: (x[0].parent.name, x[0].name))


def main():
    import sys
    finalized = None
    if "--path" in sys.argv:
        i = sys.argv.index("--path")
        if i + 1 < len(sys.argv):
            finalized = Path(sys.argv[i + 1])
            if not finalized.exists():
                print(f"Path not found: {finalized}")
                return
    if not finalized:
        for p in FINALIZED_PATHS:
            if p.exists() and (p / "night").exists() and (p / "day").exists():
                finalized = p
                break
    if not finalized:
        print("Finalized_Dark_Zurich not found. Tried:")
        for p in FINALIZED_PATHS:
            print(f"  {p}")
        return

    night_root = finalized / "night"
    day_root = finalized / "day"
    print("Scanning images...")
    pairs = find_pairs(night_root, day_root)
    print(f"Found {len(pairs)} night/day pairs in {finalized}")

    if len(pairs) < ZURICH_TRAIN + ZURICH_TEST:
        print(f"Need {ZURICH_TRAIN + ZURICH_TEST} pairs, have {len(pairs)}. Using all.")
        train_pairs = pairs[:ZURICH_TRAIN]
        test_pairs = pairs[ZURICH_TRAIN : ZURICH_TRAIN + ZURICH_TEST]
    else:
        random.seed(42)
        random.shuffle(pairs)
        train_pairs = pairs[:ZURICH_TRAIN]
        test_pairs = pairs[ZURICH_TRAIN : ZURICH_TRAIN + ZURICH_TEST]

    for name, pair_list in [("train", train_pairs), ("test", test_pairs)]:
        dst_night = OUTPUT_DIR / f"{name}_night"
        dst_day = OUTPUT_DIR / f"{name}_day"
        dst_night.mkdir(parents=True, exist_ok=True)
        dst_day.mkdir(parents=True, exist_ok=True)
        for i, (np, dp) in enumerate(pair_list):
            ext = np.suffix
            base = f"zurich_{i:05d}{ext}"
            shutil.copy2(np, dst_night / base)
            shutil.copy2(dp, dst_day / base)
        print(f"Wrote {len(pair_list)} pairs to {name}_night, {name}_day")

    # Resize to 256x256 if larger (organize_night2day also resizes output)
    total_resized = 0
    for d in (OUTPUT_DIR / "train_night", OUTPUT_DIR / "train_day",
              OUTPUT_DIR / "test_night", OUTPUT_DIR / "test_day"):
        if not d.exists():
            continue
        files = list(d.glob("*.png")) + list(d.glob("*.jpg"))
        for i, p in enumerate(files):
            if i % 200 == 0 and i > 0:
                print(f"  Resized {i}/{len(files)} in {d.name}...")
            try:
                with Image.open(p) as im:
                    im.load()
                    w, h = im.size
                    if w > MAX_SIZE or h > MAX_SIZE:
                        im = im.convert("RGB")
                        im = im.resize((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
                        im.save(p)
                        total_resized += 1
            except (OSError, ValueError) as e:
                print(f"  Skipping corrupt image {p.name}: {e}")
    if total_resized:
        print(f"Resized {total_resized} images to {MAX_SIZE}x{MAX_SIZE}")

    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
