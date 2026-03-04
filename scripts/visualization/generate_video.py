import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


_EPOCH_DIR_RE = re.compile(r"^test_(\d+)$")


def _list_epochs(progression_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in progression_dir.iterdir():
        if not p.is_dir():
            continue
        m = _EPOCH_DIR_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    h = max(left.height, right.height)
    w = left.width + right.width
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _draw_label(im: Image.Image, label: str) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 28], fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return out


def _pil_to_bgr_uint8(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create a side-by-side MP4 from CycleGAN web results across epochs.\n"
            "For each epoch folder (e.g. progression/train/test_10), it expects:\n"
            "  images/<sample>_real.png and images/<sample>_fake.png\n"
        )
    )
    ap.add_argument(
        "--progression_dir",
        default=str(Path("progression") / "train"),
        help="Folder containing epoch subfolders like test_10, test_15, ... (default: progression/train).",
    )
    ap.add_argument(
        "--sample",
        required=True,
        help="Base sample name (e.g. night_sample_1_dark_zurich or night_sample_2).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output mp4 path (default: progression_<sample>.mp4 in repo root).",
    )
    ap.add_argument("--fps", type=float, default=2.0, help="Frames per second (default: 2).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    progression_dir = Path(args.progression_dir)
    if not progression_dir.is_absolute():
        progression_dir = (project_root / progression_dir).resolve()

    epochs = _list_epochs(progression_dir)
    if not epochs:
        raise SystemExit(f"No epoch folders found under: {progression_dir}")

    out_path = Path(args.out) if args.out else (project_root / f"progression_{args.sample}.mp4")
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()

    frames: list[np.ndarray] = []
    target_size: tuple[int, int] | None = None  # (w, h)

    for epoch_num, epoch_dir in epochs:
        img_dir = epoch_dir / "images"
        real_path = img_dir / f"{args.sample}_real.png"
        fake_path = img_dir / f"{args.sample}_fake.png"
        if not real_path.exists() or not fake_path.exists():
            # Skip epochs that don't contain this sample.
            continue

        real = _read_rgb(real_path)
        fake = _read_rgb(fake_path)
        side = _side_by_side(real, fake)
        side = _draw_label(side, f"epoch {epoch_num}  |  left=real  right=fake")

        if target_size is None:
            target_size = (side.width, side.height)
        elif (side.width, side.height) != target_size:
            side = side.resize(target_size, Image.Resampling.LANCZOS)

        frames.append(_pil_to_bgr_uint8(side))

    if not frames or target_size is None:
        raise SystemExit(
            f"No frames produced. Sample '{args.sample}' not found in any epoch under {progression_dir}"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), target_size)
    if not writer.isOpened():
        raise SystemExit(
            "Failed to open video writer. If MP4 fails on your system, try installing a codec pack, "
            "or change the container/codec in this script."
        )

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    print(f"Created: {out_path}")


if __name__ == "__main__":
    main()

