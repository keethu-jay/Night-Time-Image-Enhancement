#!/usr/bin/env python3
"""
Step 3: Create "Video Progression" for each of the three weight sets (220 night->day images per video).

Reads from results/model_best_mse/test_55/images, model_best_ssim/test_70/images,
model_best_idt/test_140/images (after run_batch_inference.py). No FFmpeg required (uses imageio/cv2).

Run from project root: python scripts/make_batch_videos.py

Output: important_metrics/videos/video_best_mse.mp4, video_best_ssim.mp4, video_best_idt.mp4
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
VIDEOS_DIR = ROOT / "important_metrics" / "videos"

CONFIGS = [
    ("model_best_mse", "55"),
    ("model_best_ssim", "70"),
    ("model_best_idt", "140"),
]
FPS = 10


def collect_fake_frames(model_name: str, epoch: str) -> list[Path]:
    """Sorted list of *_fake*.png in results/<model>/test_<epoch>/images."""
    img_dir = RESULTS / model_name / f"test_{epoch}" / "images"
    if not img_dir.exists():
        return []
    paths = list(img_dir.glob("*_fake*.png")) + list(img_dir.glob("*_fake*.jpg"))
    paths.sort(key=lambda p: (p.stem.replace("_fake", "").replace("_B", ""), p.name))
    return paths


def frames_to_video(frames: list[Path], out_path: Path, fps: int = 10):
    """Write frames to MP4 with IDE/browser-friendly encoding (H.264 baseline, yuv420p)."""
    if not frames:
        print("No frames for", out_path)
        return
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
            img = imageio.imread(p)
            writer.append_data(img)
        writer.close()
        print("Wrote", out_path, f"({len(frames)} frames)")
    except Exception as e1:
        try:
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
        except Exception as e2:
            print("Failed to write video:", e1, e2)


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, epoch in CONFIGS:
        frames = collect_fake_frames(model_name, epoch)
        out = VIDEOS_DIR / f"video_best_{model_name.replace('model_best_', '')}.mp4"
        frames_to_video(frames, out, fps=FPS)
    print("Done. Videos in", VIDEOS_DIR)


if __name__ == "__main__":
    main()
