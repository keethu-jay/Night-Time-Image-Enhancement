# Scripts Overview

All scripts are meant to be run from the **project root** (e.g. `python scripts/metrics/calculate_final_grade.py`). Paths inside the scripts resolve relative to the repo root.

## preprocessing/

- **preprocessing_grid_with_mse.py** – Builds the 7-panel image (Night, HSV, Denoised, Gamma, CLAHE, Single Scale, Multi Scale) and computes MSE for each panel against the paired day image. Output: `important_metrics/preprocessing_grid_with_mse.png`. Optional args: `--night`, `--day`, `--out`.

## training/

- **sweep_epochs.py** – Sweeps all saved epoch checkpoints (one or more rounds), runs inference on the prof night image, and computes MSE (or SSIM/identity). Use it to find the best epoch per metric. Options: `--name`, `--folders`, `--metric`, `--zips`.
- **build_training_history.py** – Builds `training_history.csv` from loss logs and optionally plots error reduction.
- **plot_loss_metrics.py** – Plots each loss (G_A, cycle_A, cycle_B, idt_A, D_A) over epochs for all three rounds. Uses data from loss_log_summary. Writes PNGs to `important_metrics/`.
- **plot_cycle_idt.py** – Plots cycle_A and idt_A vs epoch from a single loss log or CSV.
- **loss_log_summary.py** – Reads loss logs for round1/2/3, computes average final values per metric, writes a summary table and CSV to `important_metrics/`.

## inference/

- **run_batch_inference.py** – Sets up model_best_mse, model_best_ssim, model_best_idt from round2 and runs test on 220 night and 220 day images. Requires `datasets/large_test` or `--dataroot`. Use `--dry-run` to print commands only.
- **run_single_night_video.py** – Runs one night image through round3 checkpoints and can build a progression video (epoch 10, 20, … 100). Default input: `important_metrics/night_gauge.png` or similar.
- **single_image_round3_video.py** – Single night image through round3 epochs to produce a short video. Expects `datasets/single_night_test/testA/night.png` or `--image`.
- **RUN_INFERENCE.md** – Step-by-step instructions for running Night→Day inference from `src/` and for running the final-grade metrics script.

## metrics/

- **run_grading.py** – **Grading script for professor/TA.** Supply your own night image (and optional day image). Runs the three best models (best MSE, best SSIM, best identity) and prints MSE/SSIM per model; saves fake-day images and a report to `important_metrics/grading/`. See README “Grading (professor/TA)”.
- **calculate_final_grade.py** – Compares real day vs fake day: channel-wise MSE, SSIM, Fidelity Score. Saves Night | Real Day | Fake Day image. Defaults look for professor pair and latest fake in results/.
- **calculate_metrics.py** – Channel-wise MSE between two images; can export loss log CSVs for plotting.
- **batch_analyzer.py** – After batch inference, computes average MSE, SSIM, and Identity Fidelity across the 220 night-to-day and 220 day-to-day results. Prints leaderboard and writes `important_metrics/batch_leaderboard.csv`.
- **generate_results_dashboard.py** – Builds model selection table, batch stats, 3-panel histograms, and console output for slides. Writes to `important_metrics/` (tables, PNGs, CSVs).
- **generate_report.py** – Parses a loss log for a given epoch and outputs a markdown table (e.g. for Slide 12).

## visualization/

- **generate_paper_figures.py** – **Paper figures and table.** Generates all figures and Table 1 for the report: Figure 1 (CycleGAN architecture), Table 1 (peak performance), Figure 2 (2x2 loss gallery), Figure 3 (Round 2 vs Round 3 qualitative), Figure 4 (generalization histograms N=440), Figure 5 (adversarial gap), Figure 6 (CLAHE vs CycleGAN). Run from project root; outputs to `paper_figures/` or `--out-dir`. Option: `--skip 1 2 ...` to skip figures by number.
- **build_important_metrics.py** – Populates `important_metrics/` with best-MSE, best-SSIM, and best-Idt_A result images and verification side-by-sides. Can run SSIM sweep and write README. Option: `--sweep-ssim`.
- **make_batch_videos.py** – Builds videos from batch inference results (220 frames per model). Output: `important_metrics/videos/video_best_mse.mp4` etc.
- **make_progression_videos.py** – Builds progression videos per round (night→day over epochs). Output: `important_metrics/videos/round1_progression.mp4` etc.
- **plot_individual_histograms.py** – One histogram per metric (MSE, SSIM, Identity Fidelity) from batch CSVs. Writes to `important_metrics/`.
- **stack_verification_images.py** – Stacks the three verification images (best MSE, SSIM, Idt_A) with labels. Output: `important_metrics/verification_stacked.png`.
- **generate_video.py** – Builds an MP4 from a progression directory of images. Used for custom progression clips; other video scripts call similar logic.
