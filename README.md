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

1. **Extract dataset** (if you have `night2day_data.zip`):
   ```
   Unzip to datasets/night2day/ with trainA, trainB, testA, testB
   ```

2. **Train** (run from project root):
   ```bash
   cd src
   python train.py --dataroot ../datasets/night2day --name night2day_model --model cycle_gan --no_html
   ```

3. **Checkpoints** save to `../checkpoints/night2day_model/`

See `night-to-day-translation/README.md` for the full data preparation workflow.
