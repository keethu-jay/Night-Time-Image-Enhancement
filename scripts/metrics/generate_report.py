#!/usr/bin/env python3
"""
Load loss_log.txt from Round 2 (or any CycleGAN training), parse the final epoch's
loss values, and optionally add channel-wise MSE (R/G/B) from generated vs real images.
Outputs a markdown table suitable for Slide 12 (presentation).
"""

import argparse
import re
from pathlib import Path

# Optional: channel-wise MSE from images (same directory as this script)
_get_channel_mse = None
_scripts_dir = Path(__file__).resolve().parent
if (_scripts_dir / "calculate_metrics.py").exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("calculate_metrics", _scripts_dir / "calculate_metrics.py")
    _mod = importlib.util.module_from_spec(_spec)
    if _spec and _spec.loader:
        _spec.loader.exec_module(_mod)
        _get_channel_mse = getattr(_mod, "get_channel_mse", None)
get_channel_mse = _get_channel_mse


LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)\s*,"
    r"\s*D_A:\s*([\d.]+),\s*G_A:\s*([\d.]+),\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*([\d.]+),\s*G_B:\s*([\d.]+),\s*cycle_B:\s*([\d.]+),\s*idt_B:\s*([\d.]+)"
)


def parse_loss_log(log_path: Path, target_epoch: int):
    """Parse loss_log.txt and return lines matching target_epoch with (D_A, G_A, ...)."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in LOSS_LINE_RE.finditer(text):
        epoch = int(m.group(1))
        if epoch != target_epoch:
            continue
        row = {
            "D_A": float(m.group(2)),
            "G_A": float(m.group(3)),
            "cycle_A": float(m.group(4)),
            "idt_A": float(m.group(5)),
            "D_B": float(m.group(6)),
            "G_B": float(m.group(7)),
            "cycle_B": float(m.group(8)),
            "idt_B": float(m.group(9)),
        }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Parse loss_log.txt and (optionally) image MSE; output markdown table for Slide 12."
    )
    ap.add_argument(
        "--log",
        default=None,
        help="Path to loss log (default: checkpoints/loss_logs/loss_log_round2.txt).",
    )
    ap.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Epoch to report (default: 100).",
    )
    ap.add_argument(
        "--gen",
        default=None,
        help="Path to generated/fake image for channel-wise MSE (optional).",
    )
    ap.add_argument(
        "--real",
        default=None,
        help="Path to ground-truth image for channel-wise MSE (optional).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write markdown to this file (default: print to stdout).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    log_path = Path(args.log) if args.log else (root / "checkpoints" / "loss_logs" / "loss_log_round2.txt")
    if not log_path.is_file():
        raise SystemExit(f"Loss log not found: {log_path}")

    lines = parse_loss_log(log_path, args.epoch)
    if not lines:
        raise SystemExit(f"No lines found for epoch {args.epoch} in {log_path}")

    # Average over all iterations in that epoch
    avg = {}
    for k in lines[0]:
        avg[k] = sum(r[k] for r in lines) / len(lines)

    md = []
    md.append(f"## Training losses (Epoch {args.epoch})")
    md.append("")
    md.append("| Loss component | Value |")
    md.append("|----------------|-------|")
    for name in ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]:
        md.append(f"| {name} | {avg[name]:.4f} |")

    if args.gen and args.real and get_channel_mse:
        gen_path = Path(args.gen)
        real_path = Path(args.real)
        if gen_path.is_file() and real_path.is_file():
            mse = get_channel_mse(gen_path, real_path)
            md.append("")
            md.append("## Channel-wise MSE (generated vs ground truth)")
            md.append("")
            md.append("| Channel | MSE |")
            md.append("|---------|-----|")
            for ch in ["R", "G", "B", "Avg"]:
                md.append(f"| {ch} | {mse[ch]:.4f} |")
        else:
            md.append("")
            md.append("*Channel MSE skipped: --gen or --real file not found.*")
    elif args.gen or args.real:
        md.append("")
        md.append("*Channel MSE: provide both --gen and --real (and ensure calculate_metrics is available).*")

    out_text = "\n".join(md)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
