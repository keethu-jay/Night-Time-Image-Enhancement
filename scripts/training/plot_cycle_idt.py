#!/usr/bin/env python3
"""
Parse loss_log.txt (or exported .csv), extract Cycle_A and Idt_A, and plot them vs Epoch.

Narrative: Cycle_A (lighting transfer) is the harder task; Idt_A (identity) shows the model
successfully preserved the building's identity throughout training.
"""
import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# CycleGAN loss line: (epoch: 35, iters: 932, ...) , D_A: ..., cycle_A: ..., idt_A: ...
_LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)\s*,"
    r"\s*D_A:\s*[\d.]+\s*,\s*G_A:\s*[\d.]+\s*,\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*[\d.]+\s*,\s*G_B:\s*[\d.]+\s*,\s*cycle_B:\s*[\d.]+\s*,\s*idt_B:\s*[\d.]+"
)


def parse_txt_log(txt_path: Path) -> list[tuple[int, float, float]]:
    """Parse loss_log.txt; return list of (epoch, cycle_A, idt_A)."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _LOSS_LINE_RE.finditer(text):
        epoch = int(m.group(1))
        cycle_a = float(m.group(2))  # lighting transfer
        idt_a = float(m.group(3))    # identity
        rows.append((epoch, cycle_a, idt_a))
    return rows


def parse_csv_log(csv_path: Path) -> list[tuple[int, float, float]]:
    """Parse exported loss_log_roundN.csv; return list of (epoch, cycle_A, idt_A)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            try:
                epoch = int(line["epoch"])
                cycle_a = float(line["cycle_A"])
                idt_a = float(line["idt_A"])
                rows.append((epoch, cycle_a, idt_a))
            except (KeyError, ValueError):
                continue
    return rows


def load_log(path: Path) -> list[tuple[int, float, float]]:
    """Load (epoch, cycle_A, idt_A) from .txt or .csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return parse_csv_log(path)
    return parse_txt_log(path)


def aggregate_by_epoch(rows: list[tuple[int, float, float]]) -> tuple[list[int], list[float], list[float]]:
    """Average cycle_A and idt_A per epoch. Returns (epochs, cycle_a_means, idt_a_means)."""
    by_epoch: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for epoch, ca, ia in rows:
        by_epoch[epoch].append((ca, ia))
    epochs = sorted(by_epoch.keys())
    cycle_means = [sum(r[0] for r in by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    idt_means = [sum(r[1] for r in by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    return epochs, cycle_means, idt_means


def main():
    ap = argparse.ArgumentParser(
        description="Plot Cycle_A (lighting transfer) and Idt_A (identity) vs Epoch from loss_log."
    )
    ap.add_argument(
        "--log",
        default=None,
        help="Path to loss_log.txt or loss_log_roundN.csv (default: checkpoints/loss_logs/loss_log_round2.txt).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output figure path (default: cycle_idt_plot.png in project root).",
    )
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive plot window.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    log_path = Path(args.log) if args.log else (root / "checkpoints" / "loss_logs" / "loss_log_round2.txt")
    out_path = Path(args.out) if args.out else (root / "cycle_idt_plot.png")

    rows = load_log(log_path)
    if not rows:
        raise SystemExit("No Cycle_A / Idt_A rows found in " + str(log_path))

    epochs, cycle_means, idt_means = aggregate_by_epoch(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, cycle_means, label=r"Cycle$_A$ (lighting transfer)", color="C0", linewidth=1.5)
    plt.plot(epochs, idt_means, label=r"Idt$_A$ (identity)", color="C1", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("Saved:", out_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
