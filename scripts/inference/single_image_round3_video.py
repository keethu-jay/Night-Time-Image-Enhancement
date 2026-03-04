#!/usr/bin/env python3
"""
Run a single night image through Round 3 weights (epochs 10,20,...,100) and create a short video.

Usage:
  python scripts/single_image_round3_video.py --image path/to/night.png
  python scripts/single_image_round3_video.py   # uses datasets/single_night_test/testA/night.png

Output: important_metrics/videos/round3_single_night_progression.mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
VIDEOS_DIR = ROOT / "important_metrics" / "videos"
SINGLE_TEST_DIR = ROOT / "datasets" / "single_night_test" / "testA"
ROUND3_NAME = "round3"
EPOCHS = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
RESULT_NAME = "round3_single"
FPS = 4


def ensure_single_image(image_path: Path | None) -> Path:
    """Ensure testA has exactly one image named night.png. Copy from image_path if given."""
    SINGLE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = SINGLE_TEST_DIR / "night.png"
    if image_path and image_path.exists():
        shutil.copy2(image_path, dest)
        print("Using image:", image_path, "->", dest)
        return dest
    if dest.exists():
        print("Using existing:", dest)
        return dest
    # Try to use first image from night2day if available
    fallback = ROOT / "datasets" / "night2day" / "testA"
    if fallback.exists():
        first = next((f for f in sorted(fallback.glob("*")) if f.suffix.lower() in (".png", ".jpg")), None)
        if first:
            shutil.copy2(first, dest)
            print("Using fallback from night2day:", first, "->", dest)
            return dest
    raise FileNotFoundError(
        "No input image. Copy your night image to datasets/single_night_test/testA/night.png "
        "or run with --image path/to/your_night.png"
    )


def run_inference(epoch: str) -> Path:
    """Run test.py for round3 at given epoch; return path to fake image."""
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot", str(SINGLE_TEST_DIR.resolve()),
        "--name", ROUND3_NAME,
        "--checkpoints_dir", str(CHECKPOINTS),
        "--results_dir", str(RESULTS),
        "--model", "test",
        "--model_suffix", "_A",
        "--no_dropout",
        "--epoch", epoch,
        "--num_test", "1",
    ]
    subprocess.run(cmd, cwd=SRC, check=True, capture_output=True, timeout=120)
    img_dir = RESULTS / ROUND3_NAME / f"test_{epoch}" / "images"
    fake = next((f for f in img_dir.glob("*_fake*") if f.suffix.lower() in (".png", ".jpg")), None)
    if not fake or not fake.exists():
        raise FileNotFoundError(f"No fake image in {img_dir}")
    return fake


def frames_to_video(frames: list[Path], out_path: Path, fps: int = 4):
    """Write frames to MP4 (H.264 baseline, yuv420p)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            str(out_path),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0"],
        )
        for p in frames:
            writer.append_data(imageio.imread(p))
        writer.close()
    except Exception as e1:
        import cv2
        out = None
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if out is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                if fourcc == -1:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            out.write(img)
        if out is not None:
            out.release()
    print("Wrote", out_path, f"({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description="Single night image through Round3 epochs -> video.")
    ap.add_argument("--image", type=Path, default=None, help="Path to your night image (default: single_night_test/testA/night.png)")
    args = ap.parse_args()

    ensure_single_image(args.image)

    frames = []
    for epoch in EPOCHS:
        print("Round3 epoch", epoch, "...", flush=True)
        try:
            frames.append(run_inference(epoch))
        except Exception as e:
            print("  Skip epoch", epoch, ":", e)

    if not frames:
        print("No frames collected.")
        sys.exit(1)

    out = VIDEOS_DIR / "round3_single_night_progression.mp4"
    frames_to_video(frames, out, fps=FPS)
    print("Done. Video:", out)


if __name__ == "__main__":
    main()
