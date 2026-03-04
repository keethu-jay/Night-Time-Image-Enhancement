#!/usr/bin/env python3
"""
Create progression videos: one per round (night→day over epochs) and one combined.
Run from project root: python scripts/make_progression_videos.py

Output:
  important_metrics/videos/round1_progression.mp4
  important_metrics/videos/round2_progression.mp4
  important_metrics/videos/round3_progression.mp4
  important_metrics/videos/all_rounds_progression.mp4
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
VIDEOS_DIR = ROOT / "important_metrics" / "videos"


def numeric_epoch_key(name: str) -> tuple[int, int]:
    """Sort key: (0, epoch) for test_50, (1, 99999) for test_latest."""
    m = re.search(r"test_(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1, 99999)


def collect_frames(round_name: str) -> list[Path]:
    """Collect night_fake.png paths for a round, sorted by epoch."""
    round_dir = RESULTS / round_name
    if not round_dir.exists():
        return []
    frames = []
    for sub in round_dir.iterdir():
        if not sub.is_dir() or not sub.name.startswith("test_"):
            continue
        img_dir = sub / "images"
        if not img_dir.exists():
            continue
        for f in img_dir.glob("*_fake*"):
            if f.suffix.lower() in (".png", ".jpg"):
                frames.append((sub.name, f))
                break
    frames.sort(key=lambda x: numeric_epoch_key(x[0]))
    return [f for _, f in frames]


def frames_to_video(frames: list[Path], out_path: Path, fps: int = 4):
    """Write frames to MP4 with IDE/browser-friendly encoding (H.264 baseline, yuv420p)."""
    if not frames:
        print("No frames for", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        # Force compatibility: baseline profile + yuv420p so players (IDE, browser) can play it
        writer = imageio.get_writer(
            str(out_path),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0"],
        )
        for p in frames:
            img = imageio.imread(p)
            writer.append_data(img)
        writer.close()
        print("Wrote", out_path, f"({len(frames)} frames)")
    except Exception as e1:
        try:
            import cv2
            h, w = None, None
            # Use avc1/H264 if available (better support than mp4v)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            if fourcc == -1:
                fourcc = cv2.VideoWriter_fourcc(*"H264")
            if fourcc == -1:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = None
            for p in frames:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if out is None:
                    h, w = img.shape[:2]
                    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                out.write(img)
            if out is not None:
                out.release()
                print("Wrote", out_path, f"({len(frames)} frames)")
        except Exception as e2:
            print("Failed to write video:", e1, e2)


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for round_name in ["round1", "round2", "round3"]:
        frames = collect_frames(round_name)
        if not frames:
            print("No frames for", round_name)
            continue
        out = VIDEOS_DIR / f"{round_name}_progression.mp4"
        frames_to_video(frames, out, fps=4)
        all_frames.extend(frames)
    if all_frames:
        frames_to_video(all_frames, VIDEOS_DIR / "all_rounds_progression.mp4", fps=6)
    print("Done. Videos in", VIDEOS_DIR)


if __name__ == "__main__":
    main()
