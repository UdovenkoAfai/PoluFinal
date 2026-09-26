from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

@dataclass
class Candidate:
    quad: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    source: str

def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)

def quad_bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    q = np.asarray(quad)
    x1 = int(np.floor(q[:, 0].min()))
    y1 = int(np.floor(q[:, 1].min()))
    x2 = int(np.ceil(q[:, 0].max()))
    y2 = int(np.ceil(q[:, 1].max()))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

def warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    q = order_quad(quad)
    tl, tr, br, bl = q
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    w = int(max(w1, w2))
    h = int(max(h1, h2))
    if w < 4 or h < 4:
        return np.empty((0, 0, 3), np.uint8)
    if h > w:
        q = np.array([bl, tl, tr, br], dtype=np.float32)
        tl, tr, br, bl = q
        w1 = np.linalg.norm(tr - tl)
        w2 = np.linalg.norm(br - bl)
        h1 = np.linalg.norm(bl - tl)
        h2 = np.linalg.norm(br - tr)
        w = int(max(w1, w2))
        h = int(max(h1, h2))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_CUBIC)

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def nms(cands: list[Candidate], threshold: float = 0.38, max_count: int = 12) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        if all(_iou(c.bbox, k.bbox) < threshold for k in kept):
            kept.append(c)
            if len(kept) >= max_count:
                break
    return kept

def classify_plate_type(crop: np.ndarray) -> tuple[str, float, dict]:
    if crop is None or crop.size == 0:
        return "other", 0.0, {}
    h, w = crop.shape[:2]
    ar = w / max(h, 1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([12, 65, 55]), np.array([45, 255, 255]))
    yellow_ratio = float(np.mean(yellow > 0))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(gray > 145))

    if 2.55 <= ar <= 6.8 and yellow_ratio > 0.12:
        typ = "type1b"
        conf = min(0.99, 0.72 + min(0.22, yellow_ratio * 0.35))
    elif 1.25 <= ar <= 2.55 and yellow_ratio < 0.20:
        typ = "type1a"
        conf = max(0.60, 0.96 - abs(ar - 1.70) * 0.22)
    elif 2.55 <= ar <= 6.8 and white_ratio > 0.30:
        typ = "type1"
        conf = max(0.58, 0.96 - abs(math.log(max(ar, 0.1) / 4.64)) * 0.35)
    else:
        typ = "other"
        conf = 0.62
    return typ, float(max(0.0, min(0.99, conf))), {"aspect": ar, "yellow_ratio": yellow_ratio, "white_ratio": white_ratio}

def _plate_quality(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if min(h, w) < 12:
        return 0.0
    ar = w / max(h, 1)
    ar1 = math.exp(-((math.log(max(ar, 0.01) / 4.64) / 0.45) ** 2))
    ar1a = math.exp(-((math.log(max(ar, 0.01) / 1.70) / 0.33) ** 2))
    ar_score = max(ar1, ar1a)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 180)
    edge_density = float(np.mean(edges > 0))
    edge_score = math.exp(-((edge_density - 0.16) / 0.13) ** 2)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (gray > 145) & (hsv[:, :, 1] < 100)
    yellow = cv2.inRange(hsv, np.array([12, 55, 50]), np.array([45, 255, 255])) > 0
    bg_ratio = float(np.mean(white | yellow))
    bg_score = min(1.0, bg_ratio / 0.48)
    dark_ratio = float(np.mean(gray < 105))
    ink_score = math.exp(-((dark_ratio - 0.18) / 0.16) ** 2)
    return float(0.36 * ar_score + 0.24 * edge_score + 0.23 * bg_score + 0.17 * ink_score)

class PlateDetector:
    def __init__(self, cascade_path: str | Path | None = None):
        self.cascade = None
        if cascade_path is not None and Path(cascade_path).exists():
            c = cv2.CascadeClassifier(str(cascade_path))
            if not c.empty():
                self.cascade = c

    def _add_rect_candidate(self, out, image, rect, score, source):
        x, y, w, h = rect
        H, W = image.shape[:2]
        pad_x = int(w * 0.06)
        pad_y = int(h * 0.14)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(W - 1, x + w + pad_x), min(H - 1, y + h + pad_y)
        q = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        crop = warp_quad(image, q)
        qscore = _plate_quality(crop)
        out.append(Candidate(q, quad_bbox(q), float(0.45 * score + 0.55 * qscore), source))

    def detect(self, image: np.ndarray) -> list[Candidate]:
        if image is None or image.size == 0:
            return []
        H0, W0 = image.shape[:2]
        max_dim = max(H0, W0)
        scale = 1.0
        work = image
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            work = cv2.resize(image, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
        H, W = work.shape[:2]
        img_area = H * W
        cands: list[Candidate] = []
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        if self.cascade is not None:
            rects = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(55, 16),
                maxSize=(int(W * 0.85), int(H * 0.45)),
            )
            for x, y, w, h in rects:
                self._add_rect_candidate(cands, work, (int(x), int(y), int(w), int(h)), 0.90, "haar")

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([12, 65, 60]), np.array([45, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)), iterations=2)
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(120, img_area * 0.00025):
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if min(rw, rh) < 8:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if not (2.0 <= ar <= 7.5):
                continue
            box = cv2.boxPoints(rect)
            crop = warp_quad(work, box)
            qscore = _plate_quality(crop)
            if qscore > 0.28:
                cands.append(Candidate(order_quad(box), quad_bbox(box), min(0.99, 0.55 + 0.44 * qscore), "yellow"))

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        white = (((val > 145) & (sat < 105)).astype(np.uint8) * 255)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=2)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(150, img_area * 0.0003) or area > img_area * 0.35:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if min(rw, rh) < 10:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if not (1.15 <= ar <= 7.2):
                continue
            rectangularity = area / max(rw * rh, 1)
            if rectangularity < 0.42:
                continue
            box = cv2.boxPoints(rect)
            crop = warp_quad(work, box)
            qscore = _plate_quality(crop)
            if qscore > 0.33:
                cands.append(Candidate(order_quad(box), quad_bbox(box), min(0.96, 0.30 + 0.66 * qscore), "bright"))

        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad = cv2.convertScaleAbs(sobel)
        _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)), iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(100, img_area * 0.00018) or area > img_area * 0.25:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if min(rw, rh) < 7:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if not (1.15 <= ar <= 8.0):
                continue
            box = cv2.boxPoints(rect)
            crop = warp_quad(work, box)
            qscore = _plate_quality(crop)
            if qscore > 0.46:
                cands.append(Candidate(order_quad(box), quad_bbox(box), min(0.90, 0.20 + 0.70 * qscore), "edges"))

        image_ar = W / max(H, 1)
        if (1.2 <= image_ar <= 6.8) and max(H, W) < 1000:
            q = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
            qscore = _plate_quality(work)
            if qscore > 0.48:
                cands.append(Candidate(q, (0, 0, W, H), min(0.92, qscore), "whole_image"))

        if scale != 1.0:
            for c in cands:
                c.quad = c.quad / scale
                c.bbox = quad_bbox(c.quad)

        cands = [c for c in cands if c.score >= 0.50]
        return nms(cands, threshold=0.36, max_count=6)
