# Midterm: Batch Inference, Metrics, and Videos

Professional-grade analysis on **440 images** (220 Night-to-Day + 220 Day-to-Day for Identity).

## Prerequisites

- **datasets/large_test**: Create `testA` (220 night images) and `testB` (220 day images). See `datasets/large_test/README.md`.
- Trained checkpoints in `checkpoints/round2`: epochs **55** (best MSE), **70** (best SSIM), **140** (best Identity).

---

## Step 1: "Best of" Batch Inference

Run **from project root**:

```powershell
python scripts/inference/run_batch_inference.py
```

This will:

1. Copy round2 weights into `checkpoints/model_best_mse`, `model_best_ssim`, `model_best_idt` (and `*_idt` for identity runs).
2. Run test **6 times** (3 models × 2: night→day and day→day identity):
   - **Night→Day (220)**: `results/model_best_mse/test_55`, `model_best_ssim/test_70`, `model_best_idt/test_140`
   - **Day→Day Identity (220)**: `results/model_best_mse_idt/test_55`, etc.

Optional: `--dry-run` to only print commands; `--dataroot path/to/large_test` to override.

---

## Step 2: Automated Metric Comparison

After Step 1:

```powershell
python scripts/metrics/batch_analyzer.py
```

- Computes **Average MSE** and **Average SSIM** over 220 night→day results per model.
- Computes **Identity Fidelity** (1 - L1) over 220 day→day results.
- Prints **Leaderboard Table** and writes `important_metrics/batch_leaderboard.csv` for Slide 31.

---

## Step 3: Video Progression (no FFmpeg)

After Step 1:

```powershell
python scripts/visualization/make_batch_videos.py
```

Output: `important_metrics/videos/video_best_mse.mp4`, `video_best_ssim.mp4`, `video_best_idt.mp4` (220 frames each, 10 fps). Use on Slide 30.

---

## Slide content (reference)

**Slide 30**: Large-Scale Generalization (N=440) — Objective, metric stability (e.g. Identity 91.4% fidelity), video evidence (A=MSE, B=SSIM, C=Identity).

**Slide 31**: Performance Leaderboard — Table from `batch_analyzer.py` / `important_metrics/batch_leaderboard.csv`.
