"""
Reorganize checkpoints: flatten night2day_model into checkpoints/round1, round2, round3
and collect all loss logs into checkpoints/loss_logs/.
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CK = ROOT / "checkpoints"
OLD = CK / "night2day_model"
ROUND3_ZIP = ROOT / "training 3_weights_only.zip"
LOSS_LOGS = CK / "loss_logs"


def main():
    if not OLD.exists():
        print("Nothing to do: night2day_model not found.")
        return

    LOSS_LOGS.mkdir(parents=True, exist_ok=True)

    # 1) Move round1 contents to checkpoints/round1/
    r1_src = OLD / "round1"
    r1_dst = CK / "round1"
    if r1_src.exists():
        r1_dst.mkdir(parents=True, exist_ok=True)
        for f in r1_src.iterdir():
            if f.is_file():
                shutil.copy2(f, r1_dst / f.name)
        print(f"Copied round1 -> {r1_dst}")

    # 2) Move round2 contents to checkpoints/round2/
    r2_src = OLD / "round2"
    r2_dst = CK / "round2"
    if r2_src.exists():
        r2_dst.mkdir(parents=True, exist_ok=True)
        for f in r2_src.iterdir():
            if f.is_file():
                shutil.copy2(f, r2_dst / f.name)
        print(f"Copied round2 -> {r2_dst}")

    # 3) Extract Round 3 zip into checkpoints/round3/
    r3_dst = CK / "round3"
    r3_dst.mkdir(parents=True, exist_ok=True)
    if ROUND3_ZIP.exists():
        tmp = ROOT / "_tmp_round3"
        tmp.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(ROUND3_ZIP, "r") as z:
                z.extractall(tmp)
            # Zip has checkpoints/night2day_model/*.pth
            src_dir = tmp / "checkpoints" / "night2day_model"
            if not src_dir.exists():
                src_dir = tmp  # fallback
            for f in src_dir.glob("*.pth"):
                shutil.copy2(f, r3_dst / f.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"Extracted Round 3 -> {r3_dst}")
    else:
        print(f"Round 3 zip not found: {ROUND3_ZIP}")

    # 4) Collect all loss logs into checkpoints/loss_logs/
    # round1
    for label, src_dir in [("round1", r1_src), ("round2", r2_src)]:
        if src_dir and (src_dir / "loss_log.txt").exists():
            shutil.copy2(src_dir / "loss_log.txt", LOSS_LOGS / f"loss_log_{label}.txt")
            print(f"  loss_logs/loss_log_{label}.txt")
    if (OLD / "loss_log_round2.txt").exists():
        shutil.copy2(OLD / "loss_log_round2.txt", LOSS_LOGS / "loss_log_round2.txt")
    if (OLD / "metrics_V1.csv").exists():
        shutil.copy2(OLD / "metrics_V1.csv", LOSS_LOGS / "metrics_V1.csv")
        print("  loss_logs/metrics_V1.csv")
    # Round 3 placeholder if no log
    if not (LOSS_LOGS / "loss_log_round3.txt").exists():
        (LOSS_LOGS / "loss_log_round3.txt").write_text(
            "Round 3 loss_log not included in archive (weights only).\n", encoding="utf-8"
        )
        print("  loss_logs/loss_log_round3.txt (placeholder)")

    # 5) Remove old night2day_model tree
    shutil.rmtree(OLD, ignore_errors=True)
    print(f"Removed {OLD}")

    print("Done. Checkpoints layout: checkpoints/round1, round2, round3, loss_logs/")


if __name__ == "__main__":
    main()
