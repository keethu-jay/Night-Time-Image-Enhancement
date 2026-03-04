# Inference: Night → Day on professor image

**Run inference from `src`.** Run the final-grade script from the **project root** (so `scripts/` is found).

**What we compare:** Real day (ground-truth `day.jpg`) vs **fake day** = model output from the night image, i.e. G(night) → `night_fake.png`. The metrics script prefers `night_fake.png` when present so we always compare real day to the fake generated from night.

---

**Correct inference command (use `testA` so only the night image is input; add `--model_suffix _A` for CycleGAN):**

```bash
cd "C:\Users\keeth\OneDrive\Desktop\DI Midterm\Night-Time-Image-Enhancement\src"
python test.py --dataroot ../datasets/prof_test/testA --name night2day_model --checkpoints_dir ../checkpoints --model test --model_suffix _A --no_dropout --num_test 1 --epoch latest --results_dir ../results
```

- **`--dataroot ../datasets/prof_test/testA`** — folder that contains **only** the night image(s). So the only output is G(night) = night_fake.png. If you use `--dataroot ../datasets/prof_test` and put both night.jpg and day.jpg there, the test will run on both and you get night_fake.png and day_fake.png; the metrics script then prefers night_fake.png.
- **`--model_suffix _A`** — **required** for CycleGAN: loads `latest_net_G_A.pth` (night→day). Without it the script looks for `latest_net_G.pth` and fails.
- **`--num_test 1`** — run on one image only (optional).
- **Output:** `results/night2day_model/test_latest/images/night_fake.png`.

From **project root** (after inference), run metrics:
```powershell
cd "C:\Users\keeth\OneDrive\Desktop\DI Midterm\Night-Time-Image-Enhancement"
python scripts/metrics/calculate_final_grade.py
```

To use `night2day_model` instead, copy the weights then run with `--name night2day_model`:

```bash
mkdir -p checkpoints/night2day_model
cp checkpoints/round3/latest_net_*.pth checkpoints/night2day_model/
cd src
python test.py --dataroot ../datasets/prof_test/testA --name night2day_model --checkpoints_dir ../checkpoints --model test --model_suffix _A --no_dropout --epoch latest --results_dir ../results
```
