#!/usr/bin/env python3
"""
Runs batch inference for the three best models (MSE, SSIM, identity). Copies round2
best-epoch weights into model_best_mse, model_best_ssim, model_best_idt, then runs
test.py twice per model: 220 night images to fake day, then 220 day images for identity.
Requires datasets/large_test/testA and testB (or pass --dataroot). Run from project root:
python scripts/inference/run_batch_inference.py. Use --dry-run to only print commands.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

# Best epochs from important_metrics (round2)
BEST_MSE_EPOCH = "55"
BEST_SSIM_EPOCH = "70"
BEST_IDT_EPOCH = "140"
SOURCE_ROUND = "round2"
NUM_TEST = 220


def setup_checkpoint(name: str, epoch: str) -> None:
    """Copy round2/{epoch}_net_G_A.pth to checkpoints/{name}/{epoch}_net_G_A.pth."""
    src = CHECKPOINTS / SOURCE_ROUND / f"{epoch}_net_G_A.pth"
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    dest_dir = CHECKPOINTS / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{epoch}_net_G_A.pth"
    shutil.copy2(src, dest)
    print("Setup checkpoint:", dest)


def run_test(
    name: str,
    epoch: str,
    dataroot: Path,
    result_name: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run src/test.py with --model test; save to results/(result_name or name)/test_{epoch}."""
    out_name = result_name or name
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot", str(dataroot),
        "--name", name,
        "--checkpoints_dir", str(CHECKPOINTS),
        "--results_dir", str(RESULTS),
        "--model", "test",
        "--model_suffix", "_A",
        "--no_dropout",
        "--epoch", epoch,
        "--num_test", str(NUM_TEST),
    ]
    # Results go to results_dir/name/phase_epoch; we want results in results/out_name
    # So we must use name=out_name and checkpoint must be in checkpoints/out_name.
    # So for identity we use a separate checkpoint copy (e.g. model_best_mse_idt) so that
    # we can pass name=model_best_mse_idt and get results there while loading same weights.
    # We already set up model_best_mse, model_best_ssim, model_best_idt. For identity we
    # use name model_best_mse_idt and need checkpoints/model_best_mse_idt with same epoch.
    if result_name:
        # Identity run: load from name (e.g. model_best_mse), save to result_name (model_best_mse_idt)
        # Test script uses opt.name for both load and save. So we need checkpoint at result_name.
        cmd[cmd.index("--name") + 1] = result_name
        # Checkpoint for identity is same epoch, already in result_name after setup
        pass
    print("Run:", " ".join(cmd))
    if not dry_run:
        r = subprocess.run(cmd, cwd=SRC, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"test.py exited with {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description="Batch inference for best MSE, SSIM, and Identity weights.")
    ap.add_argument("--dataroot", type=Path, default=ROOT / "datasets" / "large_test", help="Dataset root (testA, testB)")
    ap.add_argument("--dry-run", action="store_true", help="Only print commands, do not run.")
    args = ap.parse_args()

    dataroot = (ROOT / args.dataroot).resolve() if not args.dataroot.is_absolute() else args.dataroot.resolve()
    testA = dataroot / "testA"
    testB = dataroot / "testB"
    if not args.dry_run:
        if not testA.is_dir():
            print("ERROR: Missing", testA, "- create datasets/large_test/testA with 220 night images.")
            sys.exit(1)
        if not testB.is_dir():
            print("ERROR: Missing", testB, "- create datasets/large_test/testB with 220 day images.")
            sys.exit(1)

    configs = [
        ("model_best_mse", BEST_MSE_EPOCH),
        ("model_best_ssim", BEST_SSIM_EPOCH),
        ("model_best_idt", BEST_IDT_EPOCH),
    ]

    for name, epoch in configs:
        print("\n---", name, "epoch", epoch, "---")
        if not args.dry_run:
            setup_checkpoint(name, epoch)
            # Identity result folder: same checkpoint, different result dir
            setup_checkpoint(f"{name}_idt", epoch)

    for name, epoch in configs:
        print("\n--- Night-to-Day (220):", name, "---")
        run_test(name, epoch, testA, result_name=None, dry_run=args.dry_run)
        print("\n--- Day-to-Day Identity (220):", name, "---")
        run_test(f"{name}_idt", epoch, testB, result_name=f"{name}_idt", dry_run=args.dry_run)

    print("\nDone. Results in results/model_best_mse, model_best_ssim, model_best_idt (night->day)")
    print("and results/model_best_mse_idt, model_best_ssim_idt, model_best_idt_idt (identity).")


if __name__ == "__main__":
    main()
