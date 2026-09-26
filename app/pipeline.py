from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.detect import PlateDetector, classify_plate_type, warp_quad
from app.ocr import PlateOCR

class ANPRPipeline:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.detector = PlateDetector(model_dir / "haarcascade_russian_plate_number.xml")
        self.ocr = PlateOCR(model_dir / "char_svm.npz")

    def predict(self, image: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for cand in self.detector.detect(image):
            crop = warp_quad(image, cand.quad)
            plate_type, type_conf, diagnostics = classify_plate_type(crop)
            if plate_type == "other":
                continue
            text, ocr_conf, char_confs = self.ocr.recognize(crop, plate_type)
            hashes = text.count("#")
            valid_shape = len(text) in (8, 9)
            if (not valid_shape) or (ocr_conf < 0.565 and hashes >= 3):
                continue
            confidence = 0.30 * cand.score + 0.15 * type_conf + 0.55 * ocr_conf
            confidence *= max(0.72, 1.0 - hashes * 0.035)
            results.append({
                "plate_num": text,
                "plate_type": plate_type,
                "confidence": round(float(max(0.0, min(0.999, confidence))), 4),
                "bbox": cand.bbox,
                "quad": cand.quad,
                "source": cand.source,
                "diagnostics": diagnostics,
                "char_confidences": char_confs,
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        if not results:
            return []
        top = results[0]["confidence"]
        threshold = max(0.70, top - 0.12)
        return [r for r in results if r["confidence"] >= threshold]
