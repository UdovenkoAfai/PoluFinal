#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from app.pipeline import ANPRPipeline

SUPPORTED = {".jpg", ".jpeg", ".png"}

def draw_debug(image, results):
    vis = image.copy()
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f'{r["plate_num"]} {r["plate_type"]} {r["confidence"]:.2f}'
        cv2.putText(vis, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return vis

def main():
    p = argparse.ArgumentParser(description="Offline recognition of Russian type1/type1a/type1b license plates")
    p.add_argument("--input", required=True, type=Path, help="catalog containing .jpg/.png images")
    p.add_argument("--output", required=True, type=Path, help="semicolon-separated UTF-8 CSV")
    p.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    p.add_argument("--debug-dir", type=Path, default=None, help="optional annotated images; not used for scoring")
    p.add_argument("--max-images", type=int, default=None, help="development helper")
    args = p.parse_args()

    if not args.input.exists() or not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    model_file = args.model_dir / "char_svm.npz"
    if not model_file.exists():
        raise SystemExit("OCR model is missing. Run: python train_ocr.py")

    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if args.max_images is not None:
        files = files[: args.max_images]
    pipeline = ANPRPipeline(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    timings = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        t0 = time.perf_counter()
        results = pipeline.predict(image)
        timings.append((time.perf_counter() - t0) * 1000.0)
        for r in results:
            rows.append({
                "image": path.name,
                "plate_num": r["plate_num"],
                "plate_type": r["plate_type"],
                "confidence": f'{r["confidence"]:.4f}',
            })
        if args.debug_dir:
            cv2.imwrite(str(args.debug_dir / path.name), draw_debug(image, results))
            (args.debug_dir / f"{path.stem}.json").write_text(
                json.dumps([
                    {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
                    for r in results
                ], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "confidence"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    mean_ms = sum(timings) / len(timings) if timings else 0.0
    print(f"Processed images: {len(files)}")
    print(f"Output rows: {len(rows)}")
    print(f"Mean processing time: {mean_ms:.2f} ms/image on this machine")
    print(f"CSV: {args.output}")

if __name__ == "__main__":
    main()
