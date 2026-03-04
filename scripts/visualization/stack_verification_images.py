#!/usr/bin/env python3
"""
Stack the three best verification images (MSE, SSIM, Idt_A) into one figure with
labels: Round, Epoch, MSE, SSIM, Idt_A for each.

Run from project root: python scripts/stack_verification_images.py
Output: important_metrics/verification_stacked.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = ROOT / "important_metrics"

# Order: Best MSE, Best SSIM, Best Idt_A (same as model selection table)
VERIFICATIONS = [
    ("best_mse_verification.png", "Best MSE", "round2", "55", 0.0615, 0.2921, 0.1869),
    ("best_ssim_verification.png", "Best SSIM", "round2", "70", 0.0973, 0.3172, 0.1935),
    ("best_idt_A_verification.png", "Best Idt_A", "round2", "140", 0.0865, 0.2530, 0.1279),
]
TARGET_WIDTH = 900
LABEL_HEIGHT = 56
PAD = 8
BG = (255, 255, 255)
TEXT = (40, 40, 40)


def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


def main():
    images = []
    labels_data = []
    for fname, title, rnd, epoch, mse, ssim, idt in VERIFICATIONS:
        path = METRICS_DIR / fname
        if not path.exists():
            print("Skip (not found):", path)
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = TARGET_WIDTH / w
        new_w, new_h = TARGET_WIDTH, int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        images.append(img)
        labels_data.append((title, rnd, epoch, mse, ssim, idt))

    if not images:
        print("No verification images found in", METRICS_DIR)
        return

    font_lg = get_font(20)
    font_sm = get_font(16)
    total_h = 0
    label_strips = []
    for (title, rnd, epoch, mse, ssim, idt), img in zip(labels_data, images):
        strip = Image.new("RGB", (TARGET_WIDTH + 2 * PAD, LABEL_HEIGHT), BG)
        draw = ImageDraw.Draw(strip)
        line1 = f"{title}  |  {rnd}, Epoch {epoch}"
        line2 = f"MSE: {mse:.4f}   SSIM: {ssim:.4f}   Idt_A: {idt:.4f}"
        draw.text((PAD, 8), line1, fill=TEXT, font=font_lg)
        draw.text((PAD, 32), line2, fill=TEXT, font=font_sm)
        label_strips.append(strip)
        total_h += strip.size[1] + img.size[1] + PAD
    total_h += PAD

    out_w = TARGET_WIDTH + 2 * PAD
    out = Image.new("RGB", (out_w, total_h), BG)
    y = PAD
    for strip, img in zip(label_strips, images):
        out.paste(strip, (0, y))
        y += strip.size[1]
        out.paste(img, (PAD, y))
        y += img.size[1] + PAD

    out_path = METRICS_DIR / "verification_stacked.png"
    out.save(out_path, quality=95)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
