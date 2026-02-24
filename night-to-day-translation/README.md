# Night-to-Day Image Translation

A project for translating nighttime images to daytime using paired image-to-image translation (pix2pix-style). This document describes the complete workflow from data preparation through training.

---

## Step-by-Step Workflow

### Step 1: Pull Latest Changes from GitHub (Dark Zurich Pictures)

Your partner may have added Dark Zurich dataset images to the repository. To retrieve them:

```bash
git pull origin main
```

This fetches the Dark Zurich night/day image pairs that were added to the repo. The dataset contains paired nighttime and daytime reference images suitable for training night-to-day translation models.

**Dark Zurich Dataset Info:**
- Source: [Dark Zurich Dataset](https://github.com/dataset-ninja/dark-zurich) / [Official homepage](https://darkzurich-dataset.github.io/)
- Contains: ~2,416 nighttime images, ~2,920 twilight images, 151 annotated nighttime images with corresponding daytime references
- Used for: Semantic nighttime image segmentation and night-to-day translation

---

### Step 2: Data Layout — Professor Pair (Seed Images)

Place a single paired night/day image set in the raw folder:

```
data/raw/professor_pair/
  ├── night.jpg   # Nighttime image
  └── day.jpg     # Corresponding daytime image (same scene)
```

These serve as the seed pair for augmentation.

---

### Step 3: Creating Augmentation Functions

The script `data/augment_professor_pair.py` implements synchronized augmentation functions so both images in a pair receive the **same** transforms (required for pix2pix paired training).

| Function | Description |
|----------|-------------|
| `load_pair()` | Loads `night.jpg` and `day.jpg`, ensures same dimensions (resizes day to match night if needed) |
| `random_crop()` | Randomly crops the same region from both images |
| `random_flip_h()` | Horizontal flip (50% chance, applied to both) |
| `random_flip_v()` | Vertical flip (50% chance, applied to both) |
| `random_rotate()` | Random rotation ±15° (same angle for both) |
| `random_scale_crop()` | Random scale (0.85–1.0) then center-crop to target size |
| `apply_paired_augmentation()` | Orchestrates: scale-crop → flip H → flip V → rotate → resize to output size |

All transforms are applied identically to both images to preserve pixel-level correspondence.

---

### Step 4: Generating 400 Augmented Pairs

Run the augmentation script to generate 400 synchronized night/day pairs:

```bash
cd night-to-day-translation
python data/augment_professor_pair.py
```

**Output structure (pix2pix-style):**

```
data/processed/
  ├── trainA/   # Night images for training (220 Prof pairs)
  │   └── professor_pair_0000_night.png, professor_pair_0001_night.png, ...
  ├── trainB/   # Day images for training (paired by base name)
  │   └── professor_pair_0000_day.png, professor_pair_0001_day.png, ...
  ├── testA/    # Night images for testing (180 Prof pairs)
  └── testB/    # Day images for testing
```

- **400 total pairs** (220 train, 180 test — for combined split with Zurich)
- Filenames explicitly labeled `_night` and `_day`
- Pairs matched by base name (e.g., `professor_pair_0000_night.png` ↔ `professor_pair_0000_day.png`)

---

### Step 5: Pulling from Dark Zurich Website / Dataset

For additional paired data beyond the professor pair:

1. **Download Dark Zurich** from one of:
   - [Dataset Ninja (GitHub)](https://github.com/dataset-ninja/dark-zurich)
   - [TIB LDM Service](https://service.tib.eu/ldmservice/dataset/dark-zurich)
   - [HyperAI](https://hyper.ai/en/datasets/17466) (torrent, ~16 GB)

2. **Place Finalized_Dark_Zurich** (with `day/` and `night/` subdirs, e.g. `day/GOPR0345/`, `night/GOPR0345/`) in one of:
   - `night-to-day-translation/data/raw/dark_zurich/Finalized_Dark_Zurich`
   - `Night-Time-Image-Enhancement/dark_zurich/Finalized_Dark_Zurich`
   - `Night-Time-Image-Enhancement/Finalized_Dark_Zurich`

3. **Run prepare script** (extracts pairs, splits 900 train / 100 test, resizes to 256×256):
   ```bash
   python data/prepare_dark_zurich.py
   # Or with custom path:
   python data/prepare_dark_zurich.py --path "C:/path/to/Finalized_Dark_Zurich"
   ```

---

### Step 6: Organize Combined Dataset

Merge Zurich + Professor pairs into the final training layout:

```bash
python data/organize_night2day.py
```

**Output structure:**

```
datasets/night2day/
├── trainA/  # 1,120 total (900 Zurich Night + 220 Prof Night)
├── trainB/  # 1,120 total (900 Zurich Day + 220 Prof Day)
├── testA/   # 280 total (100 Zurich Night + 180 Prof Night)
└── testB/   # 280 total (100 Zurich Day + 180 Prof Day)
```

- Pairs use matching filenames (pix2pix): `trainA/x.png` ↔ `trainB/x.png`

---

### Step 7: Training

```bash
# Stage 1
./scripts/train_stage1.sh

# Stage 2
./scripts/train_stage2.sh
```

---

### Step 8: Inference

```bash
./scripts/run_inference.sh
```

---

## Project Structure

```
night-to-day-translation/
├── data/
│   ├── raw/
│   │   └── professor_pair/     # night.jpg, day.jpg (seed pair)
│   ├── processed/              # Augmented pairs (trainA, trainB, testA, testB)
│   ├── augment_professor_pair.py
│   ├── organize_night2day.py
│   ├── download_datasets.py
│   └── prepare_dark_zurich.py
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test.py
│   └── utils.py
├── scripts/
│   ├── setup.sh
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   └── run_inference.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_tests.ipynb
│   └── 03_results_visualization.ipynb
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `Pillow` (PIL) for augmentation; see `requirements.txt` for full list.
