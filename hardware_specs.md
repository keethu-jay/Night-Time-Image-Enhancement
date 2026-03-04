## Hardware & Environment Log

This project had **two training rounds** on different GPU pods. The main difference was available **VRAM**, which directly impacted feasible batch size and training stability.

| Feature | **Round 1: Initial Training** | **Round 2: Refinement** |
| --- | --- | --- |
| **Hardware** | NVIDIA GeForce RTX 4090 | NVIDIA H100 SXM |
| **VRAM** | 24GB GDDR6X | 80GB HBM3 |
| **Memory Bandwidth** | ~1.0 TB/s | ~3.35 TB/s |
| **Volume/Disk** | 50GB Persistent Storage | 100GB Persistent Storage |
| **Role** | Base training (150 epochs) | Polish & artifact removal (100 epochs) |
| **Performance note** | Limited to small batch size. | Enabled `batch_size 4` for more stable gradients. |

### Notes

- **Round 2** continued from **Round 1 weights** (refinement stage).
- The larger VRAM headroom on H100 makes it easier to experiment with batch size and learning rate tweaks without OOM.
