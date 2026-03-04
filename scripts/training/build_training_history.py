#!/usr/bin/env python3
"""
Parse loss logs for all three rounds (Round 1 = RTX 4090, Round 2 & 3 = H100).
Build training_history.csv with Cycle_A and Idt_A, then plot error reduction
with the H100 transition point clearly labeled.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# CycleGAN loss line in .txt logs
_LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)\s*,"
    r"\s*D_A:\s*[\d.]+\s*,\s*G_A:\s*[\d.]+\s*,\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*[\d.]+\s*,\s*G_B:\s*[\d.]+\s*,\s*cycle_B:\s*[\d.]+\s*,\s*idt_B:\s*[\d.]+"
)


def load_round1_metrics_v1(path: Path) -> list[tuple[int, float, float]]:
    """(epoch, cycle_A, idt_A) from metrics_V1.csv."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip().lstrip("L0123456789:") for ln in text.strip().splitlines() if ln.strip()]
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 10:
            continue
        try:
            epoch = int(parts[0])
            cycle_a = float(parts[3])  # Cycle_A
            idt_a = float(parts[4])    # Idt_A
            rows.append((epoch, cycle_a, idt_a))
        except (ValueError, IndexError):
            continue
    return rows


def load_txt_log(path: Path) -> list[tuple[int, float, float]]:
    """(epoch, cycle_A, idt_A) from loss_log_roundN.txt."""
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _LOSS_LINE_RE.finditer(text):
        epoch = int(m.group(1))
        cycle_a = float(m.group(2))
        idt_a = float(m.group(3))
        rows.append((epoch, cycle_a, idt_a))
    return rows


def load_csv_log(path: Path) -> list[tuple[int, float, float]]:
    """(epoch, cycle_A, idt_A) from loss_log_roundN.csv."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
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


def aggregate_by_epoch(rows: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Average cycle_A and idt_A per epoch."""
    by_epoch = defaultdict(list)
    for epoch, ca, ia in rows:
        by_epoch[epoch].append((ca, ia))
    out = []
    for e in sorted(by_epoch.keys()):
        vals = by_epoch[e]
        out.append((e, sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals)))
    return out


def main():
    root = Path(__file__).resolve().parent.parent.parent
    loss_logs = root / "checkpoints" / "loss_logs"
    if not loss_logs.exists():
        raise SystemExit("Not found: checkpoints/loss_logs")

    all_rows = []  # (round, epoch, cycle_A, idt_A)
    round1_data = []
    round2_data = []
    round3_data = []

    # Round 1 (RTX 4090): metrics_V1.csv
    m1 = loss_logs / "metrics_V1.csv"
    if m1.exists():
        round1_data = aggregate_by_epoch(load_round1_metrics_v1(m1))
        for e, ca, ia in round1_data:
            all_rows.append((1, e, ca, ia))

    # Round 2 (H100): loss_log_round2.txt or .csv
    for name in ["loss_log_round2.csv", "loss_log_round2.txt"]:
        p = loss_logs / name
        if p.exists():
            raw = load_csv_log(p) if p.suffix == ".csv" else load_txt_log(p)
            round2_data = aggregate_by_epoch(raw)
            for e, ca, ia in round2_data:
                all_rows.append((2, e, ca, ia))
            break

    # Round 3 (H100): loss_log_round3.txt or .csv
    for name in ["loss_log_round3.csv", "loss_log_round3.txt"]:
        p = loss_logs / name
        if p.exists():
            raw = load_csv_log(p) if p.suffix == ".csv" else load_txt_log(p)
            round3_data = aggregate_by_epoch(raw)
            for e, ca, ia in round3_data:
                all_rows.append((3, e, ca, ia))
            break

    if not all_rows:
        raise SystemExit("No loss data found in checkpoints/loss_logs")

    # Write training_history.csv
    out_csv = root / "training_history.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "epoch", "cycle_A", "idt_A"])
        w.writerows(all_rows)
    print("Wrote:", out_csv)

    # Build global x-axis: cumulative step so round 2 starts after round 1
    max_epoch_r1 = max(e for r, e, _, _ in all_rows if r == 1) if any(r == 1 for r, _, _, _ in all_rows) else 0
    offset_r2 = max_epoch_r1 + 1
    max_epoch_r2 = max(e for r, e, _, _ in all_rows if r == 2) if any(r == 2 for r, _, _, _ in all_rows) else 0
    offset_r3 = offset_r2 + max_epoch_r2 + 1

    x_global = []
    cycle_vals = []
    idt_vals = []
    h100_transition_x = None

    for r, e, ca, ia in all_rows:
        if r == 1:
            x_global.append(e)
        elif r == 2:
            x_global.append(offset_r2 + e)
            if h100_transition_x is None:
                h100_transition_x = offset_r2
        else:
            x_global.append(offset_r3 + e)
            if h100_transition_x is None and offset_r2 is not None:
                h100_transition_x = offset_r2
        cycle_vals.append(ca)
        idt_vals.append(ia)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_global, cycle_vals, label=r"Cycle$_A$ (lighting)", color="C0", alpha=0.8)
    ax.plot(x_global, idt_vals, label=r"Idt$_A$ (identity)", color="C1", alpha=0.8)
    if h100_transition_x is not None:
        ax.axvline(x=h100_transition_x, color="green", linestyle="--", linewidth=2, label="H100 transition (Round 2 start)")
        ax.text(h100_transition_x, ax.get_ylim()[1] * 0.95, " H100", fontsize=11, color="green", fontweight="bold")
    ax.set_xlabel("Epoch (cumulative across rounds)")
    ax.set_ylabel("Error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("Training error: RTX 4090 (Round 1) → H100 (Round 2 & 3)")
    plt.tight_layout()
    plot_path = root / "training_history_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print("Wrote:", plot_path)


if __name__ == "__main__":
    main()
