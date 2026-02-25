# Night-to-Day Image Translation

A project for translating nighttime images to daytime using CycleGAN and pix2pix. This document describes the complete workflow from data preparation through training, including technical hurdles resolved during setup.

---

## Step-by-Step Workflow

### Step 1: Pull Latest Changes from GitHub

Retrieve Dark Zurich dataset images or partner contributions:

```bash
git pull origin main
```

---

### Step 2: Seed Images — Professor Pair

Place a single paired night/day image in `data/raw/professor_pair/`:

```
data/raw/professor_pair/
  ├── night.jpg   # Nighttime image
  └── day.jpg    # Corresponding daytime image (same scene)
```

These serve as the seed pair for augmentation.

---

### Step 3: Augmentation Functions

The script `data/augment_professor_pair.py` implements **synchronized** augmentation so both images in a pair receive the same transforms (required for pix2pix).

| Function | Description |
|----------|-------------|
| `load_pair()` | Loads `night.jpg` and `day.jpg`, resizes day to match night if needed |
| `random_crop()` | Random crop same region from both |
| `random_flip_h()` / `random_flip_v()` | Horizontal/vertical flip (50% chance) |
| `random_rotate()` | Random rotation ±15° (same angle for both) |
| `random_scale_crop()` | Scale 0.85–1.0, then center-crop |
| `apply_paired_augmentation()` | Orchestrates: scale-crop → flip H → flip V → rotate → resize |

**Technical detail:** All transforms are applied identically to preserve pixel correspondence.

---

### Step 4: Generate 400 Professor Pairs

```bash
python data/augment_professor_pair.py
```

Output: `data/processed/` with 220 train pairs + 180 test pairs (split for combined Zurich + Prof dataset). Images at 256×256.

---

### Step 5: Dark Zurich — Prepare and Pair

**Place Finalized_Dark_Zurich** (with `day/` and `night/` subdirs) in:
- `data/raw/Finalized_Dark_Zurich` or
- `data/raw/dark_zurich/Finalized_Dark_Zurich`

**Technical detail:** Zurich uses different folder names for night vs day (e.g. `night/GOPR0351` vs `day/GOPR0345`). Pairing is done by **folder index + frame ID** (e.g. `GOPR0351_frame_000150` ↔ `GOPR0345_frame_000150`).

```bash
python data/prepare_dark_zurich.py
# Or: python data/prepare_dark_zurich.py --path "C:/path/to/Finalized_Dark_Zurich"
```

- Extracts 1,002 night/day pairs
- Splits: 900 train, 100 test
- Resizes images >256×256 to 256×256 (GPU stability)
- Handles corrupt/truncated images (skips with error handling)

---

### Step 6: Organize Combined Dataset

Merge Zurich + Professor into the final layout:

```bash
python data/organize_night2day.py
```

**Output structure:**

```
datasets/night2day/
├── trainA/  # 1,120 images (900 Zurich Night + 220 Prof Night)
├── trainB/  # 1,120 images (900 Zurich Day + 220 Prof Day)
├── testA/   # 280 images (100 Zurich Night + 180 Prof Night)
└── testB/   # 280 images (100 Zurich Day + 180 Prof Day)
```

**Technical detail:** Pairs use matching filenames (pix2pix convention). Output dirs are cleared before each run for a clean split.

---

### Step 7: Resize and Zip

Ensure all images ≤256×256, then zip for portability:

```bash
python data/resize_and_zip.py
```

Creates `datasets/night2day_data.zip`.

---

### Step 8: CycleGAN Setup and Training

**Repository:** Forked [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to `keethu-jay/pytorch-CycleGAN-and-pix2pix`.

**Technical hurdles resolved:**

1. **File corruption:** Core Python scripts were empty 2-byte placeholders. Rebuilt the engine using `curl` to fetch the official pytorch-CycleGAN-and-pix2pix source.
2. **Modular architecture:** Restored internal structure (`/data`, `/models`, `/options`, `/util`) and verified modules (`image_pool`, `visualizer`, `base_options`).
3. **Environment:** Installed missing deps (`wandb`), added `--no_html` for headless terminal stability.
4. **Dataset alignment:** Mapped Dark Zurich (3.72 GiB) to `trainA`/`trainB` using symbolic links (`ln -s`) where applicable.
5. **GPU:** Training runs on **Device CUDA:0** (NVIDIA RTX 3090).

**Training command:**

```bash
cd Night-Time-Image-Enhancement
conda env create -f environment.yml
conda activate pytorch-img2img
cd src
python train.py --dataroot ../datasets/night2day --name night2day_model --model cycle_gan --no_html
```

Checkpoints save to `../checkpoints/night2day_model/`.

**Current status:**
- **Hardware:** NVIDIA RTX 3090
- **Dataset:** Dark Zurich (unaligned) + Professor pairs
- **Outputs:** `./checkpoints/night2day_model/`

---

### Step 9: Inference (Testing)

Run Night-to-Day translation on new images (see pytorch-CycleGAN-and-pix2pix `test.py` and `scripts/test_*.sh`).

---

## Project Structure

```
night-to-day-translation/
├── data/
│   ├── raw/
│   │   └── professor_pair/     # night.jpg, day.jpg
│   ├── processed/              # Augmented Prof pairs
│   ├── augment_professor_pair.py
│   ├── prepare_dark_zurich.py
│   ├── organize_night2day.py
│   └── resize_and_zip.py
├── src/
├── scripts/
└── notebooks/
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `Pillow` for augmentation; see `requirements.txt` for full list.
