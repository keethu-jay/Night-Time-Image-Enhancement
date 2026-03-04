"""
Reorganize extracted progression_results: put four epoch folders in progression/train/,
remove dummy images and their HTML block.
"""
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXTRACTED = PROJECT_ROOT / "progression_results_extracted"
SRC_BASE = EXTRACTED / "content" / "pytorch-CycleGAN-and-pix2pix" / "results" / "progression" / "night2day_model"
PROGRESSION = PROJECT_ROOT / "progression"
TRAIN_DIR = PROGRESSION / "train"
EPOCH_FOLDERS = ["test_10", "test_15", "test_20", "test_25"]

DUMMY_HTML_BLOCK = re.compile(
    r'\n    <h3>dummy</h3>\n    <table border="1"[^>]*>.*?</table>',
    re.DOTALL
)


def main():
    if not SRC_BASE.exists():
        print(f"Source not found: {SRC_BASE}")
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    for name in EPOCH_FOLDERS:
        src = SRC_BASE / name
        if not src.exists():
            print(f"Skip (missing): {name}")
            continue
        dst = TRAIN_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {name} -> progression/train/{name}")

        # Delete images starting with "dummy"
        images_dir = dst / "images"
        if images_dir.exists():
            for f in list(images_dir.iterdir()):
                if f.is_file() and f.name.startswith("dummy"):
                    f.unlink()
                    print(f"  Deleted {f.name}")

        # Remove dummy section from index.html
        index_html = dst / "index.html"
        if index_html.exists():
            text = index_html.read_text(encoding="utf-8")
            new_text = DUMMY_HTML_BLOCK.sub("", text)
            if new_text != text:
                index_html.write_text(new_text, encoding="utf-8")
                print(f"  Removed dummy section from index.html")

    print("Removing extracted zip folder ...")
    shutil.rmtree(EXTRACTED, ignore_errors=True)
    print(f"Done. Epoch results are in {TRAIN_DIR}")


if __name__ == "__main__":
    main()
