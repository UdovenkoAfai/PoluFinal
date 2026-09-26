from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.constants import (
    LETTERS, DIGITS, TYPE1_SIZE, TYPE1A_SIZE,
    TYPE1_MAIN_SLOTS, TYPE1_REGION_2, TYPE1_REGION_3,
    TYPE1A_TOP, TYPE1A_BOTTOM_SERIES, TYPE1A_REGION_2, TYPE1A_REGION_3,
)
from app.char_features import hog_feature

class PlateOCR:
    def __init__(self, model_path: str | Path):
        bundle = np.load(model_path, allow_pickle=False)
        self.classes = bundle["classes"].astype(str)
        self.coef = bundle["coef"].astype(np.float32)
        self.intercept = bundle["intercept"].astype(np.float32)
        self.synthetic_val_accuracy = float(bundle["validation_accuracy"][0]) if "validation_accuracy" in bundle else None

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef.T + self.intercept[None, :]

    @staticmethod
    def _slot_quality(crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = inv.shape
        m = max(1, int(min(h, w) * 0.04))
        inv[:m, :] = 0; inv[-m:, :] = 0; inv[:, :m] = 0; inv[:, -m:] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        best = 0.0
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < max(6, 0.004 * h * w):
                continue
            hr = ch / max(h, 1)
            wr = cw / max(w, 1)
            if not (0.28 <= hr <= 0.96 and 0.06 <= wr <= 0.90):
                continue
            cx = (x + cw / 2) / max(w, 1)
            center = max(0.0, 1.0 - abs(cx - 0.5) * 1.7)
            hs = math.exp(-((hr - 0.68) / 0.30) ** 2)
            ws = math.exp(-((wr - 0.42) / 0.34) ** 2)
            fill = area / max(cw * ch, 1)
            fs = math.exp(-((fill - 0.34) / 0.28) ** 2)
            best = max(best, 0.35 * hs + 0.25 * ws + 0.20 * fs + 0.20 * center)
        return float(max(0.0, min(1.0, best)))

    def _classify_slot(self, plate: np.ndarray, box, allowed: str) -> tuple[str, float]:
        x1, y1, x2, y2 = box
        h, w = plate.shape[:2]
        variants = []
        qualities = []
        for dx in (-2, 0, 2):
            for dy in (0,):
                xa, xb = max(0, x1 + dx), min(w, x2 + dx)
                ya, yb = max(0, y1 + dy), min(h, y2 + dy)
                crop = plate[ya:yb, xa:xb]
                variants.append(hog_feature(crop))
                qualities.append(self._slot_quality(crop))
        X = np.asarray(variants, dtype=np.float32)
        scores = self._decision_function(X)
        if scores.ndim == 1:
            scores = scores[:, None]
        allowed_idx = [i for i, c in enumerate(self.classes) if c in allowed]
        sub = scores[:, allowed_idx]
        best_per_variant = sub.max(axis=1)
        combined = best_per_variant + 0.9 * np.asarray(qualities)
        vi = int(np.argmax(combined))
        ci = int(np.argmax(sub[vi]))
        chosen_idx = allowed_idx[ci]
        chosen = str(self.classes[chosen_idx])
        row = sub[vi]
        if len(row) > 1:
            best2 = np.partition(row, -2)[-2:]
            gap = float(best2[-1] - best2[-2])
        else:
            gap = 1.0
        margin_conf = 0.50 + 0.49 / (1.0 + math.exp(-2.0 * gap))
        quality = float(qualities[vi])
        conf = 0.42 * margin_conf + 0.58 * quality
        return chosen, float(min(0.995, max(0.0, conf)))

    def _read_slots(self, plate: np.ndarray, slots, allowed_seq: Iterable[str]) -> tuple[str, list[float]]:
        chars, confs = [], []
        for box, allowed in zip(slots, allowed_seq):
            ch, cf = self._classify_slot(plate, box, allowed)
            chars.append(ch)
            confs.append(cf)
        return "".join(chars), confs

    @staticmethod
    def _estimate_region_len(plate: np.ndarray, slots3) -> int:
        x1 = max(0, min(b[0] for b in slots3) - 4)
        y1 = max(0, min(b[1] for b in slots3))
        x2 = min(plate.shape[1], max(b[2] for b in slots3) + 4)
        y2 = min(plate.shape[0], max(b[3] for b in slots3))
        crop = plate[y1:y2, x1:x2]
        if crop.size == 0:
            return 2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = inv.shape
        m = max(1, int(min(h, w) * 0.03))
        inv[:m,:]=0; inv[-m:,:]=0; inv[:,:m]=0; inv[:,-m:]=0
        n, _, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if ch >= 0.42*h and cw >= 0.035*w and area >= 0.008*h*w:
                comps.append((x, cw, ch, area))
        comps.sort()
        merged=[]
        for c in comps:
            if not merged or c[0] > merged[-1][0] + merged[-1][1] * 0.75:
                merged.append(list(c))
            else:
                end=max(merged[-1][0]+merged[-1][1], c[0]+c[1])
                merged[-1][1]=end-merged[-1][0]
        return 3 if len(merged) >= 3 else 2

    def _read_region(self, plate: np.ndarray, slots2, slots3) -> tuple[str, list[float]]:
        r2, c2 = self._read_slots(plate, slots2, [DIGITS] * 2)
        r3, c3 = self._read_slots(plate, slots3, [DIGITS] * 3)
        s2 = float(np.mean(c2))
        s3 = float(np.mean(c3))
        if r3[0] not in "127":
            s3 -= 0.20
        visual_len = self._estimate_region_len(plate, slots3)
        if visual_len == 3 and s3 >= s2 - 0.10:
            return r3, c3
        if visual_len == 2 and s2 >= s3 - 0.06:
            return r2, c2
        return (r3, c3) if s3 > s2 else (r2, c2)

    def recognize(self, crop_bgr: np.ndarray, plate_type: str) -> tuple[str, float, list[float]]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "########", 0.0, []
        if plate_type == "type1a":
            plate = cv2.resize(crop_bgr, TYPE1A_SIZE, interpolation=cv2.INTER_CUBIC)
            top, ct = self._read_slots(plate, TYPE1A_TOP, [LETTERS, DIGITS, DIGITS, DIGITS])
            ser, cs = self._read_slots(plate, TYPE1A_BOTTOM_SERIES, [LETTERS, LETTERS])
            reg, cr = self._read_region(plate, TYPE1A_REGION_2, TYPE1A_REGION_3)
            text = top + ser + reg
            confs = ct + cs + cr
        else:
            plate = cv2.resize(crop_bgr, TYPE1_SIZE, interpolation=cv2.INTER_CUBIC)
            main, cm = self._read_slots(
                plate,
                TYPE1_MAIN_SLOTS,
                [LETTERS, DIGITS, DIGITS, DIGITS, LETTERS, LETTERS],
            )
            reg, cr = self._read_region(plate, TYPE1_REGION_2, TYPE1_REGION_3)
            text = main + reg
            confs = cm + cr

        out = "".join(ch if cf >= 0.57 else "#" for ch, cf in zip(text, confs))
        mean_conf = float(np.mean(confs)) if confs else 0.0
        if len(text) == 9 and text[6] not in "127":
            mean_conf *= 0.80
        return out, mean_conf, confs
