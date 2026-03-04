#!/usr/bin/env python3
"""
Reads loss logs for round1, round2, and round3 and computes average final values for
loss_G_A, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_D_A. Writes a summary table and
CSV to important_metrics/. The average is taken over the last 20% of each log. Run from
project root: python scripts/training/loss_log_summary.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS = ROOT / "checkpoints"
LOSS_LOGS = CHECKPOINTS / "loss_logs"
METRICS_DIR = ROOT / "important_metrics"
TRAINING_HISTORY_CSV = ROOT / "training_history.csv"

# Regex for .txt logs: (epoch: N, iters: M, ...) , D_A: x, G_A: x, cycle_A: x, idt_A: x, D_B: x, G_B: x, cycle_B: x, idt_B: x
_LOSS_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*(\d+).*?\)\s*,"
    r"\s*D_A:\s*([\d.]+),\s*G_A:\s*([\d.]+),\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+),"
    r"\s*D_B:\s*([\d.]+),\s*G_B:\s*([\d.]+),\s*cycle_B:\s*([\d.]+),\s*idt_B:\s*([\d.]+)"
)
CSV_COLS = ["epoch", "iteration", "D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]

# Report table: metric key, label, typical range, interpretation
REPORT_SPEC = [
    ("G_A", "loss_G_A", "0.2 - 0.5", "High realism / Adversarial success"),
    ("cycle_A", "loss_cycle_A", "0.05 - 0.15", "Information preservation (Night->Day->Night)"),
    ("cycle_B", "loss_cycle_B", "0.05 - 0.15", "Information preservation (Day->Night->Day)"),
    ("idt_A", "loss_idt_A", "0.02 - 0.08", "Geometric fidelity (The \"85% rule\")"),
    ("D_A", "loss_D_A", "oscillates", "Discriminator vs fakes (healthy if not → 0)"),
]


def parse_txt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _LOSS_RE.finditer(text):
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


def _norm_key(k: str) -> str:
    """Normalize header to our keys: Cycle_A -> cycle_A, Idt_A -> idt_A, etc."""
    m = {"cycle_a": "cycle_A", "idt_a": "idt_A", "cycle_b": "cycle_B", "idt_b": "idt_B",
         "g_a": "G_A", "d_a": "D_A", "g_b": "G_B", "d_b": "D_B"}
    return m.get(k.lower().replace(" ", ""), k)


def parse_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        key_map = {fn: _norm_key(fn) for fn in fieldnames}
        for line in r:
            try:
                row = {}
                for fn, val in line.items():
                    std = key_map.get(fn, fn)
                    if std in CSV_COLS:
                        row[std] = int(val) if std in ("epoch", "iteration") else float(val)
                if len(row) >= 8:
                    rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def load_round(round_name: str) -> list[dict]:
    """Load loss log for round1, round2, or round3. Prefer CSV then TXT in loss_logs/ or round dir."""
    base = f"loss_log_{round_name}"
    csv_path = LOSS_LOGS / f"{base}.csv"
    txt_path = LOSS_LOGS / f"{base}.txt"
    round_txt = CHECKPOINTS / round_name / "loss_log.txt"
    # Round 3: user saved as loss_log.txt (no "round3" in name)
    generic_txt = LOSS_LOGS / "loss_log.txt" if round_name == "round3" else None
    if csv_path.exists():
        rows = parse_csv(csv_path)
        if rows:
            return rows
    if txt_path.exists():
        rows = parse_txt(txt_path)
        if rows:
            return rows
    if generic_txt and generic_txt.exists():
        rows = parse_txt(generic_txt)
        if rows:
            return rows
    if round_txt.exists():
        rows = parse_txt(round_txt)
        if rows:
            return rows
    # Round 1: CSV only (no text saved in time on virtual GPU); try multiple locations
    if round_name == "round1":
        for csv_candidate in [
            LOSS_LOGS / "loss_log_round1.csv",
            LOSS_LOGS / "metrics_V1.csv",
            METRICS_DIR / "round1_loss_log.csv",
            ROOT / "round1_loss_log.csv",
            LOSS_LOGS / "round1.csv",
        ]:
            if csv_candidate.exists():
                rows = parse_csv(csv_candidate)
                if rows:
                    return rows
                if csv_candidate.name == "metrics_V1.csv":
                    rows = []
                    with open(csv_candidate, newline="", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip().lstrip("L0123456789:")
                            parts = line.split(",")
                            if len(parts) < 10:
                                continue
                            try:
                                rows.append({
                                    "epoch": int(parts[0]),
                                    "iteration": int(parts[1]),
                                    "G_A": float(parts[2]),
                                    "cycle_A": float(parts[3]),
                                    "idt_A": float(parts[4]),
                                    "D_A": float(parts[5]),
                                    "G_B": float(parts[6]),
                                    "cycle_B": float(parts[7]),
                                    "idt_B": float(parts[8]),
                                    "D_B": float(parts[9]),
                                })
                            except (ValueError, IndexError):
                                pass
                    if rows:
                        return rows
    return []


def average_final(rows: list[dict], key: str, frac: float = 0.2) -> float | None:
    """Mean of key over the last frac of rows (e.g. last 20% of training)."""
    if not rows or key not in rows[0]:
        return None
    n = max(1, int(len(rows) * frac))
    subset = rows[-n:]
    vals = [r[key] for r in subset if key in r and r[key] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    all_rows = {}
    for rnd in ["round1", "round2", "round3"]:
        rows = load_round(rnd)
        all_rows[rnd] = rows
        print(f"  {rnd}: {len(rows)} log entries")

    # Aggregate: for each metric, average "your final value" across rounds (using each round's final 20%)
    metrics_final = {}
    for key, label, _, _ in REPORT_SPEC:
        vals = []
        for rnd, rows in all_rows.items():
            if not rows:
                continue
            v = average_final(rows, key)
            if v is not None:
                vals.append((rnd, v))
        if vals:
            metrics_final[label] = {
                "per_round": dict(vals),
                "overall": sum(v for _, v in vals) / len(vals),
            }

    # Build table
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Loss Log Summary (All Three Rounds)",
        "",
        "Average final value = mean over the **last 20%** of log entries per round.",
        "",
        "## Summary Table for Report",
        "",
        "| Metric | Typical Final Range | Your Final Value | Interpretation |",
        "|--------|---------------------|------------------|----------------|",
    ]
    csv_rows = [["metric", "typical_range", "your_final_value", "interpretation"]]

    for key, label, typical, interp in REPORT_SPEC:
        if label not in metrics_final:
            md_lines.append(f"| {label} | {typical} | N/A | {interp} |")
            csv_rows.append([label, typical, "", interp])
            continue
        overall = metrics_final[label]["overall"]
        md_lines.append(f"| {label} | {typical} | {overall:.4f} | {interp} |")
        csv_rows.append([label, typical, f"{overall:.6f}", interp])

    # Per-round breakdown (optional)
    md_lines.extend([
        "",
        "## Per-round average final values",
        "",
    ])
    for rnd in ["round1", "round2", "round3"]:
        if not all_rows[rnd]:
            continue
        md_lines.append(f"### {rnd}")
        md_lines.append("")
        for key, label, typical, _ in REPORT_SPEC:
            if label not in metrics_final or rnd not in metrics_final[label]["per_round"]:
                continue
            v = metrics_final[label]["per_round"][rnd]
            md_lines.append(f"- **{label}**: {v:.4f}")
        md_lines.append("")

    out_md = METRICS_DIR / "loss_summary_table.md"
    out_csv = METRICS_DIR / "loss_summary.csv"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)
    print("Wrote", out_md)
    print("Wrote", out_csv)

    # Console (for slides) - ASCII only for Windows console
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE FOR SLIDES (copy into report)")
    print("=" * 72)
    print("\n| Metric | Typical Final Range | Your Final Value | Interpretation |")
    print("|--------|---------------------|------------------|----------------|")
    for key, label, typical, interp in REPORT_SPEC:
        interp_ascii = interp.replace("\u2192", "->").replace("\u201c", '"').replace("\u201d", '"')
        if label in metrics_final:
            overall = metrics_final[label]["overall"]
            print(f"| {label} | {typical} | {overall:.4f} | {interp_ascii} |")
        else:
            print(f"| {label} | {typical} | N/A | {interp_ascii} |")
    print("=" * 72)


if __name__ == "__main__":
    main()
