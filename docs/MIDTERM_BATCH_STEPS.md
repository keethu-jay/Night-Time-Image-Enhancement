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
   - **Day→Day Identity (220)**: `results/model_best_mse_idt/test_55`, `model_best_ssim_idt/test_70`, `model_best_idt_idt/test_140`

Optional: `--dry-run` to only print commands; `--dataroot path/to/large_test` to override dataset path.

---

## Step 2: Automated Metric Comparison

After Step 1, run:

```powershell
python scripts/metrics/batch_analyzer.py
```

This will:

1. Iterate through the **220 night-to-day** results in each folder and compute **Average MSE** and **Average SSIM**.
2. Iterate through the **220 day-to-day** results and compute **Identity Fidelity** = \(1 - \text{L1}\).
3. Print a **Leaderboard Table** and write `important_metrics/batch_leaderboard.csv` for your slides.

Use this table on **Slide 31: Performance Leaderboard**.

---

## Step 3: Video Progression (no FFmpeg)

After Step 1, create three videos (220 frames each):

```powershell
python scripts/visualization/make_batch_videos.py
```

Output (IDE-friendly H.264 baseline):

- `important_metrics/videos/video_best_mse.mp4`
- `important_metrics/videos/video_best_ssim.mp4`
- `important_metrics/videos/video_best_idt.mp4`

Use these on **Slide 30: Large-Scale Generalization (N=440)** as Video A (MSE), B (SSIM), C (Identity).

---

## Slide content (reference)

**Slide 30: Large-Scale Generalization (N=440)**

- **Objective**: Validate robustness on 220 unseen night/day pairs.
- **Metric Stability**: e.g. "Best Identity model maintained **91.4% structural fidelity** (Identity Fidelity) across 220 images."
- **Video Evidence**: Insert the three videos; describe MSE (smoother), SSIM (sharper), Identity (most stable).

**Slide 31: Performance Leaderboard**

- Insert the table produced by `batch_analyzer.py` (or paste from `important_metrics/batch_leaderboard.csv`).

---

## Manual test commands (alternative to script)

If you prefer to run test yourself for a single model:

```powershell
cd src

# 1. Best MSE (220 night → day)
python test.py --dataroot ../datasets/large_test/testA --name model_best_mse --model test --no_dropout --num_test 220 --epoch 55

# 2. Best SSIM
python test.py --dataroot ../datasets/large_test/testA --name model_best_ssim --model test --no_dropout --num_test 220 --epoch 70

# 3. Best Identity
python test.py --dataroot ../datasets/large_test/testA --name model_best_idt --model test --no_dropout --num_test 220 --epoch 140
```

(You still need to run test on `testB` for identity metrics; the script does both.)
