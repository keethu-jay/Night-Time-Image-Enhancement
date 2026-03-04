"""
Channel-wise MSE between two images, plus export of loss logs to CSV for plotting.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

# CycleGAN loss line: (epoch: 35, iters: 932, ...) , D_A: 0.122, G_A: 0.187, ...
_LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*(\d+).*?\)\s*,"
    r"\s*D_A:\s*([\d.]+),\s*G_A:\s*([\d.]+),\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*([\d.]+),\s*G_B:\s*([\d.]+),\s*cycle_B:\s*([\d.]+),\s*idt_B:\s*([\d.]+)"
)
CSV_COLUMNS = ["epoch", "iteration", "D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]


def get_channel_mse(gen_path: str | Path, real_path: str | Path) -> dict[str, float]:
    gen = np.array(Image.open(gen_path).convert("RGB"), dtype=np.float64)
    real = np.array(Image.open(real_path).convert("RGB"), dtype=np.float64)

    if gen.shape != real.shape:
        raise ValueError(f"Shape mismatch: gen={gen.shape}, real={real.shape}")

    mse_r = float(np.mean((gen[:, :, 0] - real[:, :, 0]) ** 2))
    mse_g = float(np.mean((gen[:, :, 1] - real[:, :, 1]) ** 2))
    mse_b = float(np.mean((gen[:, :, 2] - real[:, :, 2]) ** 2))
    return {"R": mse_r, "G": mse_g, "B": mse_b, "Avg": float((mse_r + mse_g + mse_b) / 3.0)}


def _parse_txt_log(txt_path: Path) -> list[dict]:
    """Parse a CycleGAN loss_log.txt; return list of dicts with keys in CSV_COLUMNS."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _LOSS_LINE_RE.finditer(text):
        rows.append({
            "epoch": int(m.group(1)),
            "iteration": int(m.group(2)),
            "D_A": float(m.group(3)),
            "G_A": float(m.group(4)),
            "cycle_A": float(m.group(5)),
            "idt_A": float(m.group(6)),
            "D_B": float(m.group(7)),
            "G_B": float(m.group(8)),
            "cycle_B": float(m.group(9)),
            "idt_B": float(m.group(10)),
        })
    return rows


def _round1_from_metrics_v1(csv_path: Path) -> list[dict]:
    """Read metrics_V1.csv (Round 1); return rows with same keys as CSV_COLUMNS."""
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    rows = []
    # Header may be "Epoch,Iteration,G_A,Cycle_A,Idt_A,D_A,G_B,Cycle_B,Idt_B,D_B" or "L1:Epoch,..."
    if not lines:
        return rows
    header = lines[0].lstrip("L0123456789:").split(",")
    for ln in lines[1:]:
        ln = ln.lstrip("L0123456789:").strip()
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) < 10:
            continue
        # metrics_V1 order: Epoch, Iteration, G_A, Cycle_A, Idt_A, D_A, G_B, Cycle_B, Idt_B, D_B
        try:
            rows.append({
                "epoch": int(parts[0]),
                "iteration": int(parts[1]),
                "D_A": float(parts[5]),
                "G_A": float(parts[2]),
                "cycle_A": float(parts[3]),
                "idt_A": float(parts[4]),
                "D_B": float(parts[9]),
                "G_B": float(parts[6]),
                "cycle_B": float(parts[7]),
                "idt_B": float(parts[8]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def export_loss_logs_to_csv(loss_logs_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """
    Create one CSV per round from checkpoints/loss_logs/ for easier plotting in matplotlib.
    - Round 1: from metrics_V1.csv (Round 1 loss data) -> loss_log_round1.csv
    - Round 2: from loss_log_round2.txt -> loss_log_round2.csv
    - Round 3: from loss_log_round3.txt -> loss_log_round3.csv
    """
    out_dir = out_dir or loss_logs_dir
    written = []

    # Round 1: metrics_V1.csv is the Round 1 loss log
    m1 = loss_logs_dir / "metrics_V1.csv"
    if m1.exists():
        rows = _round1_from_metrics_v1(m1)
        out = out_dir / "loss_log_round1.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        written.append(out)

    for round_name, txt_name in [("round2", "loss_log_round2.txt"), ("round3", "loss_log_round3.txt")]:
        txt_path = loss_logs_dir / txt_name
        if not txt_path.exists():
            continue
        rows = _parse_txt_log(txt_path)
        if not rows:
            continue
        out = out_dir / f"loss_log_{round_name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        written.append(out)

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Channel-wise MSE between two images; optionally export loss logs to CSV for plotting."
    )
    ap.add_argument("--gen", help="Path to generated/predicted image.")
    ap.add_argument("--real", help="Path to ground-truth image.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ap.add_argument(
        "--export-loss-csvs",
        action="store_true",
        help="Create CSV files for each of the three rounds in checkpoints/loss_logs/ for easier charts (matplotlib).",
    )
    args = ap.parse_args()

    if args.export_loss_csvs:
        root = Path(__file__).resolve().parent.parent.parent
        loss_logs_dir = root / "checkpoints" / "loss_logs"
        if not loss_logs_dir.exists():
            print("Not found:", loss_logs_dir)
            return
        paths = export_loss_logs_to_csv(loss_logs_dir)
        for p in paths:
            print("Wrote:", p)
        return

    if args.gen and args.real:
        metrics = get_channel_mse(args.gen, args.real)
        if args.pretty:
            print(json.dumps(metrics, indent=2))
        else:
            print(json.dumps(metrics))
        return

    ap.print_help()


if __name__ == "__main__":
    main()

