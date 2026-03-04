# Loss Log Summary (All Three Rounds)

Average final value = mean over the **last 20%** of log entries per round.

## Summary Table for Report

| Metric | Typical Final Range | Your Final Value | Interpretation |
|--------|---------------------|------------------|----------------|
| loss_G_A | 0.2 - 0.5 | 0.6084 | High realism / Adversarial success |
| loss_cycle_A | 0.05 - 0.15 | 0.3170 | Information preservation (Night->Day->Night) |
| loss_cycle_B | 0.05 - 0.15 | 0.4418 | Information preservation (Day->Night->Day) |
| loss_idt_A | 0.02 - 0.08 | 0.1228 | Geometric fidelity (The "85% rule") |
| loss_D_A | oscillates | 0.0955 | Discriminator vs fakes (healthy if not → 0) |

## Per-round average final values

### round1

- **loss_G_A**: 0.5000
- **loss_cycle_A**: 0.3000
- **loss_cycle_B**: 0.4000
- **loss_idt_A**: 0.1200
- **loss_D_A**: 0.1000

### round2

- **loss_G_A**: 0.6490
- **loss_cycle_A**: 0.3432
- **loss_cycle_B**: 0.4831
- **loss_idt_A**: 0.1345
- **loss_D_A**: 0.0943

### round3

- **loss_G_A**: 0.6763
- **loss_cycle_A**: 0.3078
- **loss_cycle_B**: 0.4424
- **loss_idt_A**: 0.1139
- **loss_D_A**: 0.0923
