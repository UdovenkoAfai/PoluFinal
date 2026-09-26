from __future__ import annotations

import cv2
import numpy as np

HOG = cv2.HOGDescriptor((32, 48), (16, 16), (8, 8), (8, 8), 9)

def normalize_char_image(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((48, 32), 255, np.uint8)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = inv.shape
    m = max(1, int(min(h, w) * 0.035))
    inv[:m, :] = 0
    inv[-m:, :] = 0
    inv[:, :m] = 0
    inv[:, -m:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    best = None
    best_score = -1.0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < max(8, 0.006 * h * w):
            continue
        if ch < 0.28 * h:
            continue
        cx = x + cw / 2
        center_pen = abs(cx - w / 2) / max(w, 1)
        full_pen = 1.0 if (cw > 0.92 * w and ch > 0.92 * h) else 0.0
        score = area + 0.8 * cw * ch - 0.7 * center_pen * area - full_pen * area
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)
    if best is None:
        ys, xs = np.where(inv > 0)
        if len(xs) == 0:
            return np.full((48, 32), 255, np.uint8)
        x, y, cw, ch = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    else:
        x, y, cw, ch = best
    glyph = inv[y:y+ch, x:x+cw]
    scale = min(26 / max(cw, 1), 42 / max(ch, 1))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((48, 32), np.uint8)
    ox, oy = (32 - nw) // 2, (48 - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = glyph
    return 255 - canvas

def hog_feature(img: np.ndarray) -> np.ndarray:
    norm = normalize_char_image(img)
    return HOG.compute(norm).reshape(-1).astype(np.float32)
