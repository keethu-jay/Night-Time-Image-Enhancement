# Model Selection Table (Best MSE, Best SSIM, Best Idt_A)

Metrics on **night prof image** (or batch avg if prof not available).

| Model | Round | Epoch | MSE (prof) | SSIM (prof) | Idt_A (train) | Flags |
|-------|-------|-------|------------|-------------|---------------|-------|
| Best MSE | round2 | 55 | 0.0615 | 0.2921 | 0.1869 | --no_dropout |
| Best SSIM | round2 | 70 | 0.0973 | 0.3172 | 0.1935 | --no_dropout |
| Best Idt_A | round2 | 140 | 0.0865 | 0.2530 | 0.1279 | --no_dropout |

---

## Batch test summary (Night→Day / Day→Day)

- N (Night→Day): 220
- N (Day→Day Identity): 220
- MSE: avg = 0.0611, std = 0.0117
- SSIM: avg = 0.3645, std = 0.0359
- Identity Fidelity: avg = 0.9801, std = 0.0032
- Success rate SSIM > 0.30: 92.7%
- Success rate Identity Fidelity > 90%: 100.0%
