# Centralized Training Command History
#
# This file is a reproducibility log: commands used to set up the environment,
# prepare data, and run the two training rounds.
#
# Notes for this repo:
# - Training entrypoints live in `src/train.py` and `src/test.py`.
# - Data prep scripts live in `night-to-day-translation/data/`.

set -euo pipefail

# --- ENVIRONMENT SETUP ---
apt-get update && apt-get install -y tmux unzip
pip install dominate visdom wandb scipy

# --- DATA PREPARATION ---
# 1. Download Zurich Dataset (example placeholder; dataset was provided separately)
python3 scripts/download_zurich.py

# 2. Augment Professor's images (Night/Day) (example placeholder)
python3 scripts/augment_data.py --input ./raw --output ./datasets/night2day

# 3. Create Train/Test Split (80/20) (example placeholder)
python3 scripts/split_data.py --dataroot ./datasets/night2day

# --- DATA PREPARATION (actual scripts in this repository) ---
# Professor pair augmentation + Zurich prep + final dataset organization:
#
# python night-to-day-translation/data/augment_professor_pair.py
# python night-to-day-translation/data/prepare_dark_zurich.py --path "/path/to/Finalized_Dark_Zurich"
# python night-to-day-translation/data/organize_night2day.py
# python night-to-day-translation/data/resize_and_zip.py

# --- TRAINING ROUNDS ---
# Round 1: 4090 Pod (150 Epochs)
python3 src/train.py --dataroot ./datasets/night2day --name night2day_model \
  --model cycle_gan --batch_size 1 --n_epochs 100 --n_epochs_decay 50

# Round 2: H100 Pod (100 Epoch Refinement)
# We used --continue_train to load the 4090 weights and --no_dropout to fix artifacts
python3 src/train.py --dataroot ./datasets/night2day --name night2day_model \
  --model cycle_gan --continue_train --lr 0.00005 --batch_size 4 \
  --no_dropout --n_epochs 50 --n_epochs_decay 50

