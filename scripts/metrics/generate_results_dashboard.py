#!/usr/bin/env python3
"""
Generate the final Results Dashboard for slides:
  1. Model Selection Table (Round, Epoch, MSE, SSIM, Idt_A, Flags) + Prof-image comparison
  2. Batch statistics (N=440) and Success Rate (SSIM>0.30, Identity Fidelity>90%)
  3. 3-panel histograms -> metric_distributions.png
  4. Console output for copy-paste into PowerPoint

Run from project root: python scripts/generate_results_dashboard.py
"""
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS = ROOT / "checkpoints"
LOSS_LOGS = CHECKPOINTS / "loss_logs"
RESULTS = ROOT / "results"
TRAINING_HISTORY_CSV = ROOT / "training_history.csv"
METRICS_DIR = ROOT / "important_metrics"
# Test set: prefer large_test, fallback night2day
DATAROOT = ROOT / "datasets" / "large_test"
DATAROOT_FALLBACK = ROOT / "datasets" / "night2day"
PROF_TEST = ROOT / "datasets" / "prof_test"
SIZE = (256, 256)

# Known best (from important_metrics / sweep)
BEST_MSE = ("round2", "55")
BEST_SSIM = ("round2", "70")
BEST_IDT = ("round2", "140")
FLAGS = "--no_dropout"


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64) / 255.0


def mse_channelwise_avg(real: np.ndarray, fake: np.ndarray) -> float:
    mse_r = float(np.mean((real[..., 0] - fake[..., 0]) ** 2))
    mse_g = float(np.mean((real[..., 1] - fake[..., 1]) ** 2))
    mse_b = float(np.mean((real[..., 2] - fake[..., 2]) ** 2))
    return (mse_r + mse_g + mse_b) / 3.0


def ssim_one(real: np.ndarray, fake: np.ndarray) -> float:
    if ssim_func is None:
        return float("nan")
    r = np.mean(real, axis=-1) if real.ndim == 3 else real
    f = np.mean(fake, axis=-1) if fake.ndim == 3 else fake
    return float(ssim_func(r, f, data_range=1.0))


def l1_one(real: np.ndarray, fake: np.ndarray) -> float:
    return float(np.mean(np.abs(real.astype(np.float64) - fake.astype(np.float64))))


def collect_fake_paths(images_dir: Path) -> list[Path]:
    paths = list(images_dir.glob("*_fake*.png")) + list(images_dir.glob("*_fake*.jpg"))
    paths.sort(key=lambda p: (p.stem.replace("_fake", "").replace("_B", ""), p.name))
    return paths


# ----- 1. Model selection: find best epochs and Idt_A from logs -----
def load_idt_a_for_epoch(round_name: str, epoch: int) -> float | None:
    """Get idt_A for (round, epoch) from training_history.csv or loss_logs."""
    if TRAINING_HISTORY_CSV.exists():
        with open(TRAINING_HISTORY_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["round"]) == int(round_name.replace("round", "")) and int(row["epoch"]) == epoch:
                        return float(row["idt_A"])
                except (ValueError, KeyError):
                    continue
    csv_path = LOSS_LOGS / f"loss_log_{round_name}.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["epoch"]) == epoch:
                        return float(row["idt_A"])
                except (ValueError, KeyError):
                    continue
    return None


def get_prof_metrics(round_name: str, epoch: str) -> tuple[float, float]:
    """MSE and SSIM on the prof night image: fake from results/round/test_epoch vs real from prof_test/testB."""
    fake_dir = RESULTS / round_name / f"test_{epoch}" / "images"
    if not fake_dir.exists():
        return float("nan"), float("nan")
    fakes = list(fake_dir.glob("*_fake*.png")) + list(fake_dir.glob("*_fake*.jpg"))
    if not fakes:
        return float("nan"), float("nan")
    real_dir = PROF_TEST / "testB"
    if not real_dir.exists():
        real_dir = ROOT / "night-to-day-translation" / "data" / "raw" / "professor_pair"
        real_candidates = list(real_dir.glob("day*.jpg")) + list(real_dir.glob("*.png")) if real_dir.exists() else []
    else:
        real_candidates = sorted(real_dir.glob("*"))
    real_candidates = [p for p in real_candidates if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not real_candidates:
        return float("nan"), float("nan")
    real_path = real_candidates[0]
    fake_path = fakes[0]
    real = load_rgb(real_path)
    fake = load_rgb(fake_path)
    return mse_channelwise_avg(real, fake), ssim_one(real, fake)


def build_model_selection_table() -> list[dict]:
    """Rows: Best MSE, Best SSIM, Best Idt_A with Round, Epoch, MSE (prof), SSIM (prof), Idt_A (training), Flags."""
    rows = []
    for label, (rnd, epoch) in [("Best MSE", BEST_MSE), ("Best SSIM", BEST_SSIM), ("Best Idt_A", BEST_IDT)]:
        ep_int = int(epoch)
        idt_a = load_idt_a_for_epoch(rnd, ep_int)
        mse_prof, ssim_prof = get_prof_metrics(rnd, epoch)
        rows.append({
            "Model": label,
            "Round": rnd,
            "Epoch": epoch,
            "MSE (prof)": mse_prof,
            "SSIM (prof)": ssim_prof,
            "Idt_A (train)": idt_a,
            "Flags": FLAGS,
        })
    return rows


# ----- 2. Batch stats and per-image lists for histograms -----
def get_test_dataroot() -> Path:
    if (DATAROOT / "testB").exists():
        return DATAROOT
    if (DATAROOT_FALLBACK / "testB").exists():
        return DATAROOT_FALLBACK
    return DATAROOT


def collect_per_image_metrics(results_dir: Path, testB_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """Use model_best_mse (night->day) and model_best_mse_idt (day->day). Return (mses, ssims, idt_fidelities)."""
    mses, ssims, idt_fids = [], [], []
    # Night-to-day: one model is enough for distribution of 220 images; use model_best_mse
    img_dir = results_dir / "model_best_mse" / "test_55" / "images"
    if img_dir.exists() and testB_dir.exists():
        fakes = collect_fake_paths(img_dir)
        reals = sorted(testB_dir.glob("*"))
        reals = [p for p in reals if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")][: len(fakes)]
        for f, r in zip(fakes, reals):
            real = load_rgb(r)
            fake = load_rgb(f)
            mses.append(mse_channelwise_avg(real, fake))
            ssims.append(ssim_one(real, fake))
    # Day-to-day identity
    idt_dir = results_dir / "model_best_mse_idt" / "test_55" / "images"
    if idt_dir.exists() and testB_dir.exists():
        fakes = collect_fake_paths(idt_dir)
        reals = sorted(testB_dir.glob("*"))
        reals = [p for p in reals if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")][: len(fakes)]
        for f, r in zip(fakes, reals):
            real = load_rgb(r)
            fake = load_rgb(f)
            idt_fids.append(1.0 - l1_one(real, fake))
    return mses, ssims, idt_fids


# ----- 3. Histograms -----
def save_histograms(mses: list[float], ssims: list[float], idt_fidelities: list[float], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("seaborn")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if mses:
        axes[0].hist(mses, bins=min(30, len(mses)), color="steelblue", edgecolor="white")
        axes[0].set_title("MSE (Night->Day, N={})".format(len(mses)))
        axes[0].set_xlabel("MSE")
    else:
        axes[0].set_title("MSE (no data)")
    if ssims:
        axes[1].hist(ssims, bins=min(30, len(ssims)), color="seagreen", edgecolor="white")
        axes[1].set_title("SSIM (Night->Day, N={})".format(len(ssims)))
        axes[1].set_xlabel("SSIM")
    else:
        axes[1].set_title("SSIM (no data)")
    if idt_fidelities:
        axes[2].hist(idt_fidelities, bins=min(30, len(idt_fidelities)), color="coral", edgecolor="white")
        axes[2].set_title("Identity Fidelity (Day->Day, N={})".format(len(idt_fidelities)))
        axes[2].set_xlabel("Identity Fidelity")
    else:
        axes[2].set_title("Identity Fidelity (no data)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def main():
    # ---- 1. Model Selection Table (with prof-image comparison) ----
    table_rows = build_model_selection_table()
    md_lines = [
        "| Model | Round | Epoch | MSE (prof) | SSIM (prof) | Idt_A (train) | Flags |",
        "|-------|-------|-------|------------|-------------|---------------|-------|",
    ]
    for r in table_rows:
        mse_s = f"{r['MSE (prof)']:.4f}" if not np.isnan(r["MSE (prof)"]) else "N/A"
        ssim_s = f"{r['SSIM (prof)']:.4f}" if not np.isnan(r["SSIM (prof)"]) else "N/A"
        idt_s = f"{r['Idt_A (train)']:.4f}" if r["Idt_A (train)"] is not None else "N/A"
        md_lines.append(f"| {r['Model']} | {r['Round']} | {r['Epoch']} | {mse_s} | {ssim_s} | {idt_s} | {r['Flags']} |")
    model_table_md = "\n".join(md_lines)

    # ---- 2. Batch statistics ----
    dataroot = get_test_dataroot()
    testB = dataroot / "testB"
    mses, ssims, idt_fids = collect_per_image_metrics(RESULTS, testB)
    n_ntd = len(mses)
    n_idt = len(idt_fids)
    avg_mse = float(np.mean(mses)) if mses else float("nan")
    std_mse = float(np.std(mses)) if mses else float("nan")
    avg_ssim = float(np.mean(ssims)) if ssims else float("nan")
    std_ssim = float(np.std(ssims)) if ssims else float("nan")
    avg_idt = float(np.mean(idt_fids)) if idt_fids else float("nan")
    std_idt = float(np.std(idt_fids)) if idt_fids else float("nan")
    pct_ssim_30 = 100.0 * sum(1 for s in ssims if s > 0.30) / len(ssims) if ssims else 0.0
    pct_idt_90 = 100.0 * sum(1 for i in idt_fids if i > 0.90) / len(idt_fids) if idt_fids else 0.0

    # ---- 3. Histograms ----
    out_fig = METRICS_DIR / "metric_distributions.png"
    save_histograms(mses, ssims, idt_fids, out_fig)

    # ---- 3b. CSV data for histograms (for slides / Excel) ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = []
    if mses:
        path_mse = METRICS_DIR / "histogram_mse_night_to_day.csv"
        with open(path_mse, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_index", "mse"])
            for i, v in enumerate(mses, start=1):
                w.writerow([i, round(v, 6)])
        csv_files.append(path_mse)
    if ssims:
        path_ssim = METRICS_DIR / "histogram_ssim_night_to_day.csv"
        with open(path_ssim, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_index", "ssim"])
            for i, v in enumerate(ssims, start=1):
                w.writerow([i, round(v, 6)])
        csv_files.append(path_ssim)
    if idt_fids:
        path_idt = METRICS_DIR / "histogram_identity_fidelity_day_to_day.csv"
        with open(path_idt, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_index", "identity_fidelity"])
            for i, v in enumerate(idt_fids, start=1):
                w.writerow([i, round(v, 6)])
        csv_files.append(path_idt)
    for p in csv_files:
        print("Wrote", p)

    # ---- 4. Save table to file ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = METRICS_DIR / "model_selection_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Model Selection Table (Best MSE, Best SSIM, Best Idt_A)\n\n")
        f.write("Metrics on **night prof image** (or batch avg if prof not available).\n\n")
        f.write(model_table_md)
        f.write("\n\n---\n\n## Batch test summary (Night→Day / Day→Day)\n\n")
        f.write(f"- N (Night→Day): {n_ntd}\n")
        f.write(f"- N (Day→Day Identity): {n_idt}\n")
        f.write(f"- MSE: avg = {avg_mse:.4f}, std = {std_mse:.4f}\n")
        f.write(f"- SSIM: avg = {avg_ssim:.4f}, std = {std_ssim:.4f}\n")
        f.write(f"- Identity Fidelity: avg = {avg_idt:.4f}, std = {std_idt:.4f}\n")
        f.write(f"- Success rate SSIM > 0.30: {pct_ssim_30:.1f}%\n")
        f.write(f"- Success rate Identity Fidelity > 90%: {pct_idt_90:.1f}%\n")
    print("Wrote", table_path)

    # ---- 5. Console output for slides ----
    print("\n" + "=" * 72)
    print("  RESULTS DASHBOARD — copy below into PowerPoint")
    print("=" * 72)
    print("\n--- Model Selection Table (Slide 31: Final Leaderboard) ---\n")
    print(model_table_md)
    print("\n--- Batch statistics (N=440 / test set) ---")
    print(f"  Night->Day: N = {n_ntd}")
    print(f"  MSE:  avg = {avg_mse:.4f},  std = {std_mse:.4f}")
    print(f"  SSIM: avg = {avg_ssim:.4f},  std = {std_ssim:.4f}")
    print(f"  Day->Day Identity: N = {n_idt}")
    print(f"  Identity Fidelity: avg = {avg_idt:.4f},  std = {std_idt:.4f}")
    print("\n--- Success rate (Slide 32: Generalization) ---")
    print(f"  % images with SSIM > 0.30:        {pct_ssim_30:.1f}%")
    print(f"  % images with Identity Fid. > 90%: {pct_idt_90:.1f}%")
    print("\n--- Files generated ---")
    print(f"  {out_fig}")
    print(f"  {table_path}")
    for p in csv_files:
        print(f"  {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
