## Technical Project Log — Night-Time Image Enhancement (Night → Day)

This project performs **night-to-day image translation** using a CycleGAN-style image-to-image translation pipeline and a custom dataset built from **Dark Zurich** plus a paired “professor” example augmented into many synchronized pairs.

### What was done (high level)

- **Data pipeline**
  - Seeded a paired sample (`night.jpg` ↔ `day.jpg`) and generated synchronized augmentations.
  - Prepared the **Finalized_Dark_Zurich** night/day pairs into a consistent train/test layout.
  - Merged both sources into a final `datasets/night2day/` split.
- **Training (two rounds)**
  - **Round 1**: baseline training run (limited VRAM → small batch).
  - **Round 2**: refinement run continuing from Round 1 checkpoints with a smaller LR and higher batch size on a larger GPU.
- **Evaluation artifacts**
  - Saved epoch snapshots into `progression/train/test_*/` and removed `dummy*` images.
  - Provided scripts to compute channel-wise MSE and generate side-by-side MP4s across epochs.

### Hardware

See **`hardware_specs.md`** at the repo root for the two pod specs (RTX 4090 vs H100 SXM) and notes on VRAM-driven differences.

### Reproducibility commands

See **`commands.sh`** at the repo root. It includes the environment setup and the two training commands used for **Round 1** and **Round 2**.

### Repository structure (organized)

```
Night-Time-Image-Enhancement/
  checkpoints/                 # Exported weights + logs (flat; no night2day_model nesting)
    round1/                    # Round 1 weights (.pth)
    round2/                    # Round 2 weights (.pth)
    round3/                    # Round 3 weights (.pth)
    loss_logs/                 # All loss metric files in one place
      loss_log_round1.txt, loss_log_round2.txt, loss_log_round3.txt, metrics_V1.csv
  src/                         # CycleGAN / pix2pix code (training/inference run on virtual GPU)
  night-to-day-translation/    # Data preparation only (scripts + seed images)
  datasets/night2day/          # Final dataset split (trainA/trainB/testA/testB)
  progression/train/           # Epoch result snapshots (test_10, test_15, test_20, test_25)
  scripts/
    calculate_metrics.py       # Channel-wise MSE (R/G/B + Avg)
    generate_video.py          # MP4 across epochs (side-by-side real vs fake)
    generate_report.py         # Parse loss_logs for Slide 12 table
  docs/
    TECHNICAL_PROJECT_LOG.md   # (this file)
```

### How the CycleGAN flags map to what we did

The core training/testing entrypoints are:

- `src/train.py`: training loop, checkpoint saving
- `src/test.py`: inference + HTML/web output
- `src/options/*`: defines CLI flags and defaults

Flags used in the two-round workflow:

- **`--continue_train`**: used in Round 2 to resume from Round 1 checkpoints and continue optimizing.
- **`--no_dropout`**: used in Round 2 to reduce visible artifacts (dropout disabled).
- **`--no_html`**: used during training in headless environments to skip HTML generation and avoid I/O overhead.
- **`--batch_size`**: Round 1 was constrained by VRAM; Round 2 used a larger batch size on H100 for more stable gradients.
- **`--lr`**: reduced in Round 2 for refinement.

### Utility scripts (quick usage)

- **Channel-wise MSE**

```bash
python scripts/metrics/calculate_metrics.py --gen "path/to/fake.png" --real "path/to/real.png" --pretty
```

- **Epoch progression video**

```bash
python scripts/visualization/generate_video.py --sample night_sample_1_dark_zurich --progression_dir progression/train --fps 1
```

