#!/usr/bin/env python3
"""
Run inference once per epoch (one at a time to avoid memory issues), compute MSE/SSIM
for each, then pick the best and save final_verification.png.

Modes:
  - Single folder: python scripts/sweep_epochs.py --name round3 [--metric mse]
  - Multiple folders (round1, round2, round3 .pth files): python scripts/sweep_epochs.py --folders round1 round2 round3 --metric mse
  - Zips: python scripts/sweep_epochs.py --zips
"""
import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def discover_epochs(checkpoint_dir: Path) -> list[str]:
    """Find all epoch numbers that have a G_A checkpoint file present (and optionally 'latest')."""
    pattern = re.compile(r"^(\d+)_net_G_A\.pth$")
    epochs = []
    if not checkpoint_dir.exists():
        return epochs
    for f in checkpoint_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            epochs.append(m.group(1))
    if (checkpoint_dir / "latest_net_G_A.pth").exists():
        epochs.append("latest")
    def _key(e):
        if e == "latest":
            return (1, 99999)
        return (0, int(e))
    # Only return epochs that actually exist on disk
    def exists(e):
        if e == "latest":
            return (checkpoint_dir / "latest_net_G_A.pth").exists()
        return (checkpoint_dir / f"{e}_net_G_A.pth").exists()
    return sorted([e for e in set(epochs) if exists(e)], key=_key)


def find_ckpt_folder(extract_root: Path) -> Path | None:
    """Return the folder that contains *_net_G_A.pth (any depth under extract_root)."""
    if not extract_root.exists():
        return None
    # Any file matching *_net_G_A.pth → its parent dir is the checkpoint folder
    for f in extract_root.rglob("*_net_G_A.pth"):
        if f.is_file():
            return f.parent
    # Fallback: case-insensitive or alternate naming (e.g. net_G_A.PTH)
    for f in extract_root.rglob("*.pth"):
        if f.is_file() and "net_G_A" in f.name:
            return f.parent
    return None


def ensure_flat_ckpt_folder(extract_root: Path, target_folder: Path) -> Path:
    """
    Copy every *.pth (and *.txt) under extract_root into target_folder by filename.
    This picks up all epochs no matter how deeply nested (e.g. 150 epochs in subdirs).
    """
    target_folder.mkdir(parents=True, exist_ok=True)
    copied_ga = 0
    for f in extract_root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() == ".pth" or f.suffix.lower() == ".txt":
            shutil.copy2(f, target_folder / f.name)
            if "net_G_A" in f.name:
                copied_ga += 1
    if copied_ga == 0:
        raise FileNotFoundError(f"No *_net_G_A.pth found under {extract_root}")
    return target_folder


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extract zip to out_dir. Returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def run_inference(root: Path, name: str, epoch: str, checkpoints_dir: Path, results_dir: Path) -> Path:
    """Run test.py for one epoch. Returns path to the fake image."""
    src = root / "src"
    fake_dir = results_dir / name / f"test_{epoch}" / "images"
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot", str(root / "datasets" / "prof_test" / "testA"),
        "--name", name,
        "--checkpoints_dir", str(checkpoints_dir),
        "--model", "test",
        "--model_suffix", "_A",
        "--no_dropout",
        "--epoch", epoch,
        "--results_dir", str(results_dir),
    ]
    result = subprocess.run(cmd, cwd=src, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"exit code {result.returncode}")
    fake_path = next((f for f in fake_dir.glob("*_fake*") if f.suffix.lower() in (".png", ".jpg")), None)
    if not fake_path or not fake_path.exists():
        raise FileNotFoundError(f"Inference did not produce a fake image under {fake_dir}")
    return fake_path


def run_metrics_quiet(root: Path, fake_path: Path) -> tuple[float, float]:
    """Run calculate_final_grade.py --fake <path> --quiet; return (mse, ssim)."""
    cmd = [
        sys.executable,
        str(root / "scripts" / "metrics" / "calculate_final_grade.py"),
        "--fake", str(fake_path),
        "--quiet",
    ]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    mse_m = re.search(r"MSE_avg=([\d.]+)", result.stdout) or re.search(r"MSE=([\d.]+)", result.stdout)
    ssim_m = re.search(r"SSIM=([\d.]+)", result.stdout)
    if not mse_m or not ssim_m:
        raise ValueError(f"Could not parse metrics from: {result.stdout}")
    return float(mse_m.group(1)), float(ssim_m.group(1))


def sweep_folder(
    root: Path,
    checkpoints_dir: Path,
    results_dir: Path,
    name: str,
    ckpt_folder: Path,
) -> list[tuple[str, float, float]]:
    """Run sweep for one checkpoint folder. Returns list of (epoch, mse, ssim)."""
    epochs = discover_epochs(ckpt_folder)
    if not epochs:
        return []
    results_list = []
    total = len(epochs)
    for i, epoch in enumerate(epochs):
        print(f"  [{i+1}/{total}] Epoch {epoch} ... ", end="", flush=True)
        try:
            fake_path = run_inference(root, name, epoch, checkpoints_dir, results_dir)
            mse_val, ssim_val = run_metrics_quiet(root, fake_path)
            results_list.append((epoch, mse_val, ssim_val))
            print(f"MSE={mse_val:.4f} SSIM={ssim_val:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
    return results_list


def main():
    ap = argparse.ArgumentParser(
        description="Sweep epoch checkpoints (folder or every zip), one epoch at a time, pick best by SSIM."
    )
    ap.add_argument("--name", default=None, help="Single folder: model name under checkpoints_dir (e.g. round3)")
    ap.add_argument("--folders", nargs="+", default=None, help="Multiple folders to sweep (e.g. round1 round2 round3). Uses .pth files in each.")
    ap.add_argument("--zips", nargs="*", default=None, help="Zip paths (or glob). Unzip each, sweep all epochs, pick best across all.")
    ap.add_argument("--metric", choices=("mse", "ssim"), default="ssim", help="Pick best epoch by: mse (lower is better) or ssim (higher is better)")
    ap.add_argument("--checkpoints", default=None, help="Checkpoints root (default: project_root/checkpoints)")
    ap.add_argument("--results", default=None, help="Results root (default: project_root/results)")
    ap.add_argument("--keep-extracted", action="store_true", help="Do not delete extracted zip folders after sweep (for --zips)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    checkpoints_dir = Path(args.checkpoints) if args.checkpoints else (root / "checkpoints")
    results_dir = Path(args.results) if args.results else (root / "results")

    all_results: list[tuple[str, str, str, float, float]] = []  # (source_label, epoch, mse, ssim)

    if args.zips is not None:
        # Resolve paths and globs to a list of .zip files
        zip_paths: list[Path] = []
        to_scan = list(args.zips) if args.zips else []
        if not to_scan:
            # Default: all .zip files under checkpoints and project root
            zip_paths = list(checkpoints_dir.glob("*.zip")) + list(root.glob("*.zip"))
        else:
            for p in to_scan:
                path = Path(p)
                if not path.is_absolute():
                    path = (root / path).resolve()
                if path.is_file() and path.suffix.lower() == ".zip":
                    zip_paths.append(path)
                elif path.is_dir():
                    zip_paths.extend(sorted(path.glob("*.zip")))
                else:
                    for f in root.glob(str(p)):
                        if f.is_file() and f.suffix.lower() == ".zip":
                            zip_paths.append(f)
        zip_paths = sorted(set(zip_paths))
        if not zip_paths:
            print("No zip files found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(zip_paths)} zip(s). Will extract, sweep each, then pick best across all.\n")

        extracted_folders: list[tuple[str, Path]] = []  # (label, path to folder used for inference)

        for zip_idx, zip_path in enumerate(zip_paths):
            if not zip_path.exists():
                print(f"Skip (missing): {zip_path}")
                continue
            # Sanitize label for paths (no spaces) to avoid loader issues
            label = zip_path.stem.replace(" ", "_")
            extract_root = checkpoints_dir / "_sweep_extract" / label
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"Zip [{zip_idx+1}/{len(zip_paths)}] {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    pth_names = [n for n in zf.namelist() if ".pth" in n.lower() and "net_G_A" in n]
                print(f"  (zip contains {len(pth_names)} G_A .pth files)")
                extract_zip(zip_path, extract_root)
            except Exception as e:
                print(f"  Extract FAILED: {e}")
                continue
            # Flatten so that *_net_G_A.pth live in a single-named folder for --name
            flat_name = f"_sweep_{label}"
            flat_folder = checkpoints_dir / flat_name
            try:
                ensure_flat_ckpt_folder(extract_root, flat_folder)
            except FileNotFoundError as e:
                # Show first few paths inside zip to help debug
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        names = [n for n in zf.namelist() if ".pth" in n.lower()][:8]
                        if names:
                            print(f"  (zip has .pth files: {names[0]} ...)")
                        else:
                            print(f"  (zip has no .pth files)")
                except Exception:
                    pass
                print(f"  {e}")
                continue
            extracted_folders.append((label, flat_folder))
            epochs = discover_epochs(flat_folder)
            n_ga = len(list(flat_folder.glob("*_net_G_A.pth")))
            print(f"  Epochs: {len(epochs)} ({n_ga} G_A checkpoints) {epochs[:5]}{'...' if len(epochs) > 5 else ''}")
            for i, epoch in enumerate(epochs):
                ckpt_file = flat_folder / ("latest_net_G_A.pth" if epoch == "latest" else f"{epoch}_net_G_A.pth")
                if not ckpt_file.exists():
                    print(f"  [{i+1}/{len(epochs)}] Epoch {epoch} ... SKIP (missing {ckpt_file.name})")
                    continue
                print(f"  [{i+1}/{len(epochs)}] Epoch {epoch} ... ", end="", flush=True)
                try:
                    fake_path = run_inference(root, flat_name, epoch, checkpoints_dir, results_dir)
                    mse_val, ssim_val = run_metrics_quiet(root, fake_path)
                    all_results.append((label, epoch, mse_val, ssim_val))
                    print(f"MSE={mse_val:.4f} SSIM={ssim_val:.4f}")
                except Exception as e:
                    err = str(e).split("\n")[0][:80]
                    print(f"FAILED: {err}")
            if not args.keep_extracted:
                shutil.rmtree(extract_root, ignore_errors=True)
            print()

        # Best across all zips
        if not all_results:
            print("No successful runs from any zip.", file=sys.stderr)
            sys.exit(1)
        if args.metric == "mse":
            best = min(all_results, key=lambda x: x[2])
        else:
            best = max(all_results, key=lambda x: x[3])
        best_label, best_epoch, best_mse, best_ssim = best[0], best[1], best[2], best[3]
        print(f"Best (across all zips, by {args.metric.upper()}): {best_label} epoch {best_epoch}  MSE={best_mse:.6f}  SSIM={best_ssim:.6f}  Fidelity={best_ssim*100:.2f}%")

        # Ensure best checkpoint folder still exists (we may have deleted extract_root but flat_folder remains)
        flat_name = f"_sweep_{best_label}"
        best_ckpt = checkpoints_dir / flat_name
        if not best_ckpt.exists():
            # Re-extract the winning zip (match by sanitized stem)
            zip_path = next((z for z in zip_paths if z.stem.replace(" ", "_") == best_label), None)
            if zip_path and zip_path.exists():
                extract_root = checkpoints_dir / "_sweep_extract" / best_label
                extract_root.mkdir(parents=True, exist_ok=True)
                extract_zip(zip_path, extract_root)
                ensure_flat_ckpt_folder(extract_root, best_ckpt)
                if not args.keep_extracted:
                    shutil.rmtree(extract_root, ignore_errors=True)

        print(f"\nRe-running inference for best ({best_label} epoch {best_epoch}) and saving final_verification.png ...")
        fake_path = run_inference(root, flat_name, best_epoch, checkpoints_dir, results_dir)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "metrics" / "calculate_final_grade.py"),
                "--fake", str(fake_path),
            ],
            cwd=root,
            check=True,
        )
        if not args.keep_extracted:
            for _label, flat_path in extracted_folders:
                if flat_path.exists():
                    shutil.rmtree(flat_path, ignore_errors=True)
            sweep_extract = checkpoints_dir / "_sweep_extract"
            if sweep_extract.exists():
                shutil.rmtree(sweep_extract, ignore_errors=True)
        print("Done. final_verification.png is for the best epoch across all zips.")

    elif args.folders:
        # Multiple folders (round1, round2, round3): sweep each, pick best by metric across all
        all_results: list[tuple[str, str, float, float]] = []  # (folder, epoch, mse, ssim)
        for folder_name in args.folders:
            ckpt_folder = checkpoints_dir / folder_name
            if not ckpt_folder.exists():
                print(f"Skip (missing): {ckpt_folder}")
                continue
            epochs = discover_epochs(ckpt_folder)
            if not epochs:
                print(f"No epochs in {folder_name}; skip.")
                continue
            print(f"\n--- {folder_name} ({len(epochs)} epochs) ---")
            for i, epoch in enumerate(epochs):
                ckpt_file = ckpt_folder / ("latest_net_G_A.pth" if epoch == "latest" else f"{epoch}_net_G_A.pth")
                if not ckpt_file.exists():
                    print(f"  [{i+1}/{len(epochs)}] Epoch {epoch} ... SKIP (missing)")
                    continue
                print(f"  [{i+1}/{len(epochs)}] Epoch {epoch} ... ", end="", flush=True)
                try:
                    fake_path = run_inference(root, folder_name, epoch, checkpoints_dir, results_dir)
                    mse_val, ssim_val = run_metrics_quiet(root, fake_path)
                    all_results.append((folder_name, epoch, mse_val, ssim_val))
                    print(f"MSE={mse_val:.4f} SSIM={ssim_val:.4f}")
                except Exception as e:
                    err = str(e).split("\n")[0][:80]
                    print(f"FAILED: {err}")
        if not all_results:
            print("No successful runs.", file=sys.stderr)
            sys.exit(1)
        if args.metric == "mse":
            best = min(all_results, key=lambda x: x[2])
            print(f"\nBest (by MSE, lower is better): {best[0]} epoch {best[1]}  MSE={best[2]:.6f}  SSIM={best[3]:.6f}")
        else:
            best = max(all_results, key=lambda x: x[3])
            print(f"\nBest (by SSIM): {best[0]} epoch {best[1]}  MSE={best[2]:.6f}  SSIM={best[3]:.6f}  Fidelity={best[3]*100:.2f}%")
        best_folder, best_epoch = best[0], best[1]
        print(f"\nRe-running inference for best ({best_folder} epoch {best_epoch}) and saving final_verification.png ...")
        fake_path = run_inference(root, best_folder, best_epoch, checkpoints_dir, results_dir)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "metrics" / "calculate_final_grade.py"),
                "--fake", str(fake_path),
            ],
            cwd=root,
            check=True,
        )
        print("Done. final_verification.png is for the best epoch.")

    elif args.name:
        # Single folder mode
        ckpt_folder = checkpoints_dir / args.name
        epochs = discover_epochs(ckpt_folder)
        if not epochs:
            print(f"No epoch checkpoints found in {ckpt_folder} (looking for *_net_G_A.pth)", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(epochs)} epochs in {args.name}: {epochs[:5]}{'...' if len(epochs) > 5 else ''}\n")
        results_list = sweep_folder(root, checkpoints_dir, results_dir, args.name, ckpt_folder)
        if not results_list:
            print("No successful runs.", file=sys.stderr)
            sys.exit(1)
        if args.metric == "mse":
            best = min(results_list, key=lambda x: x[1])
        else:
            best = max(results_list, key=lambda x: x[2])
        best_epoch, best_mse, best_ssim = best[0], best[1], best[2]
        print(f"\nBest epoch (by {args.metric.upper()}): {best_epoch}  MSE={best_mse:.6f}  SSIM={best_ssim:.6f}  Fidelity={best_ssim*100:.2f}%")
        print(f"\nRe-running inference for best epoch {best_epoch} and saving final_verification.png ...")
        fake_path = run_inference(root, args.name, best_epoch, checkpoints_dir, results_dir)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "metrics" / "calculate_final_grade.py"),
                "--fake", str(fake_path),
            ],
            cwd=root,
            check=True,
        )
        print("Done. final_verification.png and printed metrics are for the best epoch.")
    else:
        print("Use --name <folder>, --folders round1 round2 round3, or --zips", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
