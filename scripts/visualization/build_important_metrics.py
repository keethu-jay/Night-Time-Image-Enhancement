#!/usr/bin/env python3
"""
Fills important_metrics/ with the best-MSE, best-SSIM, and best-Idt_A result images and
verification side-by-sides, then writes a README there. Assumes sweep has already been
run so that results/roundN/test_*/images/night_fake.png exist. With --sweep-ssim it
runs an SSIM sweep to find and save the best SSIM epoch. Run from project root:
python scripts/visualization/build_important_metrics.py
"""
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = ROOT / "important_metrics"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
LOSS_LOGS = CHECKPOINTS / "loss_logs"
TRAINING_HISTORY_CSV = ROOT / "training_history.csv"

BEST_MSE_ROUND = "round2"
BEST_MSE_EPOCH = "55"


def ensure_dir():
    METRICS_DIR.mkdir(exist_ok=True)
    print("Created", METRICS_DIR)


def copy_best_mse():
    """Copy best-MSE epoch result (round2 epoch 55) and a verification image."""
    fake = RESULTS / BEST_MSE_ROUND / f"test_{BEST_MSE_EPOCH}" / "images" / "night_fake.png"
    if not fake.exists():
        # Try night_fake.*
        img_dir = RESULTS / BEST_MSE_ROUND / f"test_{BEST_MSE_EPOCH}" / "images"
        if img_dir.exists():
            for f in img_dir.glob("*_fake*"):
                fake = f
                break
    if not fake.exists():
        print("Warning: best MSE image not found at", fake, "(run sweep with --metric mse first)")
        return
    dest_fake = METRICS_DIR / "best_mse_round2_epoch55_fake.png"
    shutil.copy2(fake, dest_fake)
    print("Saved", dest_fake)
    # Verification (night | real day | fake day)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "metrics" / "calculate_final_grade.py"),
            "--fake", str(fake),
            "--out", str(METRICS_DIR / "best_mse_verification.png"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if (METRICS_DIR / "best_mse_verification.png").exists():
        print("Saved best_mse_verification.png (Night | Real Day | Fake Day for best MSE)")


def find_best_ssim():
    """Run sweep with --metric ssim and save best-SSIM image into important_metrics."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "training" / "sweep_epochs.py"),
            "--folders", "round1", "round2", "round3",
            "--metric", "ssim",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600000,
    )
    # Parse "Best (by SSIM): round2 epoch 70 ..." to get folder and epoch
    best_ssim_folder = best_ssim_epoch = None
    for line in result.stdout.splitlines():
        if "Best (by SSIM)" in line or "Best (across" in line and "SSIM" in line:
            # round2 epoch 70
            m = re.search(r"round\d+\s+epoch\s+(\w+)", line, re.I)
            if m:
                best_ssim_epoch = m.group(1)
            m2 = re.search(r"(round\d+)\s+epoch", line, re.I)
            if m2:
                best_ssim_folder = m2.group(1)
            break
    if best_ssim_folder and best_ssim_epoch:
        fake = RESULTS / best_ssim_folder / f"test_{best_ssim_epoch}" / "images" / "night_fake.png"
        if not fake.exists():
            for f in (RESULTS / best_ssim_folder / f"test_{best_ssim_epoch}" / "images").glob("*_fake*"):
                fake = f
                break
        if fake.exists():
            shutil.copy2(fake, METRICS_DIR / f"best_ssim_{best_ssim_folder}_epoch{best_ssim_epoch}_fake.png")
            shutil.copy2(ROOT / "final_verification.png", METRICS_DIR / "best_ssim_verification.png")
            print("Saved best_ssim_verification.png and fake image for", best_ssim_folder, "epoch", best_ssim_epoch)
    return best_ssim_folder, best_ssim_epoch


def load_idt_a_all_rounds():
    """Load (round, epoch, idt_A) from training_history.csv and loss_logs. Return list and best (min idt_A)."""
    rows = []  # (round, epoch, idt_A)
    if TRAINING_HISTORY_CSV.exists():
        with open(TRAINING_HISTORY_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for line in r:
                try:
                    rows.append((int(line["round"]), int(line["epoch"]), float(line["idt_A"])))
                except (ValueError, KeyError):
                    continue
    # If training_history only has round 2, try loss_logs for round1 and round3
    _LOSS_RE = re.compile(
        r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)\s*,"
        r"\s*D_A:\s*[\d.]+\s*,\s*G_A:\s*[\d.]+\s*,\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    )
    for rnd, fname in [(1, "loss_log_round1.txt"), (3, "loss_log_round3.txt")]:
        if any(x[0] == rnd for x in rows):
            continue
        p = LOSS_LOGS / fname
        if not p.exists():
            p = LOSS_LOGS / fname.replace(".txt", ".csv")
            if p.exists():
                with open(p, newline="", encoding="utf-8") as f:
                    rd = csv.DictReader(f)
                    for line in rd:
                        try:
                            rows.append((rnd, int(line["epoch"]), float(line["idt_A"])))
                        except (ValueError, KeyError):
                            continue
            continue
        if p.suffix == ".txt":
            text = p.read_text(encoding="utf-8", errors="replace")
            by_epoch = {}
            for m in _LOSS_RE.finditer(text):
                e, _, idt = int(m.group(1)), float(m.group(2)), float(m.group(3))
                by_epoch[e] = by_epoch.get(e, []) + [idt]
            for e, vals in by_epoch.items():
                rows.append((rnd, e, sum(vals) / len(vals)))
    if not rows:
        return [], None
    best = min(rows, key=lambda x: x[2])
    return rows, best


def copy_best_idt_a():
    """Find epoch with best (lowest) idt_A and copy its fake image. Only consider epochs that have checkpoint and result."""
    rows, _ = load_idt_a_all_rounds()
    if not rows:
        print("Warning: no idt_A data found in training_history.csv or loss_logs")
        return None
    # Only consider (round, epoch) for which we have a result image (from sweep)
    best = None
    best_idt = float("inf")
    for rnd, epoch, idt_val in rows:
        folder = f"round{rnd}"
        fake = RESULTS / folder / f"test_{epoch}" / "images" / "night_fake.png"
        if not fake.exists():
            img_dir = RESULTS / folder / f"test_{epoch}" / "images"
            if img_dir.exists():
                for f in img_dir.glob("*_fake*"):
                    fake = f
                    break
        if fake.exists() and idt_val < best_idt:
            best_idt = idt_val
            best = (rnd, epoch, idt_val, fake)
    if not best:
        print("Warning: no idt_A result image found (sweep may not have run for idt_A epochs)")
        return None
    rnd, epoch, idt_val, fake = best
    folder = f"round{rnd}"
    shutil.copy2(fake, METRICS_DIR / f"best_idt_A_round{rnd}_epoch{epoch}_fake.png")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "metrics" / "calculate_final_grade.py"),
            "--fake", str(fake),
            "--out", str(METRICS_DIR / "best_idt_A_verification.png"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    print("Saved best_idt_A images (round{} epoch {}, idt_A={:.4f})".format(rnd, epoch, idt_val))
    return (rnd, epoch, idt_val)


def write_readme(best_ssim_folder=None, best_ssim_epoch=None, best_idt=None):
    """Write README.md in important_metrics linking to all images. best_idt = (round, epoch, idt_val) if saved."""
    idt_line = ""
    if best_idt:
        rnd, epoch, idt_val = best_idt
        idt_line = f"Best **Idt_A** (identity loss, lower is better): **round{rnd} epoch {epoch}** (idt_A = {idt_val:.4f}).\n- [best_idt_A_round{rnd}_epoch{epoch}_fake.png](best_idt_A_round{rnd}_epoch{epoch}_fake.png) — fake day output for that epoch.\n- [best_idt_A_verification.png](best_idt_A_verification.png) — Night | Real Day | Fake Day.\n\n"
    ssim_line = ""
    if best_ssim_folder and best_ssim_epoch:
        ssim_line = f"Best **SSIM** (structural similarity, higher is better): **{best_ssim_folder} epoch {best_ssim_epoch}**.\n- [best_ssim_{best_ssim_folder}_epoch{best_ssim_epoch}_fake.png](best_ssim_{best_ssim_folder}_epoch{best_ssim_epoch}_fake.png) — fake day output.\n- [best_ssim_verification.png](best_ssim_verification.png) — Night | Real Day | Fake Day.\n\n"
    readme = f"""# Important metrics and result images

This folder holds the **best** result images from sweeping all epoch weights across rounds 1–3.

## Best MSE (channel-wise, lower is better)

Best **MSE** (reportable metric): **round2 epoch 55** (lowest channel-wise average MSE).

- [best_mse_round2_epoch55_fake.png](best_mse_round2_epoch55_fake.png) — fake day image produced by the model at that epoch.
- [best_mse_verification.png](best_mse_verification.png) — side-by-side: **Night** (input) | **Real Day** (ground truth) | **Fake Day** (model output), with MSE and Similarity % in the bar.

## Best SSIM (structural similarity)

{ssim_line or "Run `python scripts/build_important_metrics.py --sweep-ssim` to populate best SSIM images.\\n\\n"}

## Best Idt_A (identity loss from training)

{idt_line or "No idt_A data found.\\n\\n"}

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
"""
    (METRICS_DIR / "README.md").write_text(readme, encoding="utf-8")
    print("Wrote", METRICS_DIR / "README.md")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-ssim", action="store_true", help="Run full SSIM sweep to find and save best SSIM (slow)")
    args = ap.parse_args()
    ensure_dir()
    copy_best_mse()
    best_ssim_folder, best_ssim_epoch = None, None
    if args.sweep_ssim:
        best_ssim_folder, best_ssim_epoch = find_best_ssim()
    else:
        # Try to copy existing best SSIM (round2 epoch 70 from earlier runs)
        for folder, epoch in [("round2", "70"), ("round2", "50")]:
            fake = RESULTS / folder / f"test_{epoch}" / "images" / "night_fake.png"
            if not fake.exists():
                d = RESULTS / folder / f"test_{epoch}" / "images"
                if d.exists():
                    for f in d.glob("*_fake*"):
                        fake = f
                        break
            if fake.exists():
                shutil.copy2(fake, METRICS_DIR / f"best_ssim_{folder}_epoch{epoch}_fake.png")
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "metrics" / "calculate_final_grade.py"), "--fake", str(fake),
                     "--out", str(METRICS_DIR / "best_ssim_verification.png")],
                    cwd=ROOT, capture_output=True, check=False,
                )
                best_ssim_folder, best_ssim_epoch = folder, epoch
                print("Saved best_ssim images (using existing", folder, "epoch", epoch, ")")
                break
    copy_best_idt_a()
    write_readme(best_ssim_folder=best_ssim_folder, best_ssim_epoch=best_ssim_epoch)
    print("Done. See important_metrics/README.md for links to all images.")


if __name__ == "__main__":
    main()
