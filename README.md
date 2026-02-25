# Night-Time Image Enhancement

Night-to-day image translation using CycleGAN. Data preparation pipeline + training code.

## Repository Layout

```
/datasets/night2day/     # trainA, trainB, testA, testB (extract from zip locally)
/src/                    # CycleGAN/pix2pix Python logic (train.py, test.py, data/, models/, options/, util/)
/checkpoints/            # .pth files and loss_log.txt
/night-to-day-translation/  # Data prep scripts (augment, prepare_dark_zurich, organize, resize_and_zip)
```

## Quick Start

1. **Extract dataset** (unzip `night2day_data.zip` to `datasets/night2day/` with trainA, trainB, testA, testB).  
   Do not keep the .zip in the repo—it's too large for GitHub.

2. **Restore checkpoints** (if resuming): Place `.pth` files in `checkpoints/night2day_model/` so the script finds them.

3. **Train** (run from project root):
   ```bash
   cd src
   python train.py --dataroot ../datasets/night2day --name night2day_model --model cycle_gan --no_html
   ```

4. **Checkpoints** save to `checkpoints/night2day_model/`

See `night-to-day-translation/README.md` for the full data preparation workflow.
