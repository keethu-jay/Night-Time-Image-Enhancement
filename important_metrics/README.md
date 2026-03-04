# Important metrics and result images

This folder holds the **best** result images from sweeping all epoch weights across rounds 1–3.

## Best MSE (channel-wise, lower is better)

Best **MSE** (reportable metric): **round2 epoch 55** (lowest channel-wise average MSE).

- [best_mse_round2_epoch55_fake.png](best_mse_round2_epoch55_fake.png) — fake day image produced by the model at that epoch.
- [best_mse_verification.png](best_mse_verification.png) — side-by-side: **Night** (input) | **Real Day** (ground truth) | **Fake Day** (model output), with MSE and Similarity % in the bar.

## Best SSIM (structural similarity)

Best **SSIM** (structural similarity, higher is better): **round2 epoch 70**.
- [best_ssim_round2_epoch70_fake.png](best_ssim_round2_epoch70_fake.png) — fake day output.
- [best_ssim_verification.png](best_ssim_verification.png) — Night | Real Day | Fake Day.



## Best Idt_A (identity loss from training)

No idt_A data found.\n\n

## Progression videos

In the project root, run:

```bash
python scripts/visualization/make_progression_videos.py
```

This creates:

- `important_metrics/videos/round1_progression.mp4` — night→day progression over round 1 epochs.
- `important_metrics/videos/round2_progression.mp4` — round 2 epochs.
- `important_metrics/videos/round3_progression.mp4` — round 3 epochs.
- `important_metrics/videos/all_rounds_progression.mp4` — all three rounds in sequence (full training progression).
