#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.constants import ALL_CHARS
from app.char_features import hog_feature
from generator.plate_render import FONT_CANDIDATES


def render_char(ch: str, rng: random.Random) -> np.ndarray:
    w, h = 56, 80
    bgv = rng.randint(225, 255)
    if rng.random() < 0.25:
        bg = (rng.randint(230, 250), rng.randint(175, 215), rng.randint(20, 60))
    else:
        bg = (bgv, bgv, bgv)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    font_path = rng.choice(FONT_CANDIDATES) if FONT_CANDIDATES else None
    size = rng.randint(48, 67)
    font = ImageFont.truetype(font_path, size=size) if font_path else ImageFont.load_default()
    bb = d.textbbox((0, 0), ch, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (w - tw) / 2 - bb[0] + rng.uniform(-3.5, 3.5)
    y = (h - th) / 2 - bb[1] + rng.uniform(-4.0, 4.0)
    d.text((x, y), ch, font=font, fill=(rng.randint(0, 25),) * 3)
    angle = rng.uniform(-5.0, 5.0)
    im = im.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=bg)
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.7)))
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(1, 8), arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        k = rng.choice([1, 2])
        arr = cv2.erode(arr, np.ones((k, k), np.uint8), iterations=1)
    return arr


def main():
    p = argparse.ArgumentParser(description="Train lightweight HOG+SVM OCR character classifier")
    p.add_argument("--samples-per-char", type=int, default=260)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--out", type=Path, default=ROOT / "models" / "char_svm.npz")
    args = p.parse_args()

    rng = random.Random(args.seed)
    X, y = [], []
    for ch in ALL_CHARS:
        for _ in range(args.samples_per_char):
            X.append(hog_feature(render_char(ch, rng)))
            y.append(ch)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    idx = np.arange(len(y))
    np.random.default_rng(args.seed).shuffle(idx)
    cut = int(len(idx) * 0.9)
    tr, va = idx[:cut], idx[cut:]

    model = LinearSVC(C=4.0, dual="auto", max_iter=8000, random_state=args.seed)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    acc = accuracy_score(y[va], pred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, classes=model.classes_.astype("U1"), coef=model.coef_.astype(np.float32), intercept=model.intercept_.astype(np.float32), validation_accuracy=np.array([acc], dtype=np.float32))
    print(f"Saved {args.out}")
    print(f"Synthetic character validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
