#!/usr/bin/env python3
"""
Run a single night image through every round3 checkpoint and create a progression video.

Shows: night input → fake at epoch 10 → epoch 20 → ... → latest, so you see how
round3 training improved on that one picture.

Usage:
  python scripts/run_single_night_video.py [path_to_night_image.png]

Output: important_metrics/videos/round3_single_night_progression.mp4
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
VIDEOS_DIR = ROOT / "important_metrics" / "videos"
SINGLE_TEST_DIR = ROOT / "datasets" / "single_night_gauge" / "testA"


def find_night_image() -> Path | None:
    candidates = [
        ROOT / "important_metrics" / "night_gauge.png",
        ROOT / "important_metrics" / "night_gauge.jpg",
        Path(r"C:\Users\keeth\.cursor\projects\c-Users-keeth-OneDrive-Desktop-DI-Midterm-Night-Time-Image-Enhancement"
             r"\assets\c__Users_keeth_AppData_Roaming_Cursor_User_workspaceStorage_b0868737524cf5dadf6490ebec246715_images_night_sample_1_dark_zurich-e5d6164a-4f08-4e76-aa1e-9e5b445797db.png"),
        next((ROOT / "datasets" / "night2day" / "testA").glob("*.png"), None) if (ROOT / "datasets" / "night2day" / "testA").exists() else None,
    ]
    for p in candidates:
        if p and p.exists():
            return p
    return None


def get_round3_epochs() -> list[str]:
    """All numeric epoch checkpoints in round3, sorted (e.g. ['10','20',...,'100'])."""
    r3 = CHECKPOINTS / "round3"
    if not r3.exists():
        return []
    nums = []
    for p in r3.glob("*_net_G_A.pth"):
        try:
            n = int(p.stem.split("_")[0])
            nums.append(n)
        except ValueError:
            pass
    return [str(n) for n in sorted(set(nums))]


def main():
    if len(sys.argv) > 1:
        night_path = Path(sys.argv[1]).resolve()
    else:
        night_path = find_night_image()

    if not night_path or not night_path.exists():
        print("No night image found. Put gauge at important_metrics/night_gauge.png or pass path.")
        sys.exit(1)

    SINGLE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = SINGLE_TEST_DIR / "night.png"
    shutil.copy2(night_path, dest)
    print("Using night image:", night_path.name)

    epochs = get_round3_epochs()
    if not epochs:
        print("No round3 checkpoints found in", CHECKPOINTS / "round3")
        sys.exit(1)
    print("Round3 progression: epochs", epochs)

    # Run test.py for each epoch; collect [night, fake_epoch1, fake_epoch2, ...]
    frame_paths = [dest]
    for epoch in epochs:
        cmd = [
            sys.executable,
            "test.py",
            "--dataroot", str(SINGLE_TEST_DIR.resolve()),
            "--name", "round3",
            "--checkpoints_dir", str(CHECKPOINTS.resolve()),
            "--results_dir", str(RESULTS.resolve()),
            "--model", "test",
            "--model_suffix", "_A",
            "--no_dropout",
            "--epoch", epoch,
            "--num_test", "1",
        ]
        r = subprocess.run(cmd, cwd=SRC, timeout=120, capture_output=True, text=True)
        if r.returncode != 0:
            print("test.py epoch", epoch, "failed:", r.stderr or r.stdout)
            continue
        fake_dir = RESULTS / "round3" / f"test_{epoch}" / "images"
        fakes = sorted(fake_dir.glob("*_fake*.png")) + sorted(fake_dir.glob("*_fake*.jpg"))
        if fakes:
            frame_paths.append(fakes[0])
            print("  epoch", epoch, "->", fakes[0].name)

    if len(frame_paths) < 2:
        print("Need at least one fake output.")
        sys.exit(1)

    # Video: one frame per step (night then each epoch); hold each ~0.5s so progression is visible
    fps = 4
    frames = []
    for p in frame_paths:
        frames.extend([p] * 2)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_video = VIDEOS_DIR / "round3_single_night_progression.mp4"
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            str(out_video),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0"],
        )
        for p in frames:
            writer.append_data(imageio.imread(p))
        writer.close()
        print("Wrote", out_video, "(" + str(len(frames)) + " frames, round3 progression)")
    except Exception as e:
        import cv2
        out = None
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if out is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
            out.write(img)
        if out is not None:
            out.release()
            print("Wrote", out_video)
    print("Done. Video shows round3 training progression on this image.")

if __name__ == "__main__":
    main()
