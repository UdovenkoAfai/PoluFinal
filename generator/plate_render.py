from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from app.constants import (
    LETTERS,
    DIGITS,
    TYPE1_SIZE,
    TYPE1A_SIZE,
    TYPE1_MAIN_SLOTS,
    TYPE1_REGION_2,
    TYPE1_REGION_3,
    TYPE1A_TOP,
    TYPE1A_BOTTOM_SERIES,
    TYPE1A_REGION_2,
    TYPE1A_REGION_3,
)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
]
FONT_CANDIDATES = [p for p in FONT_CANDIDATES if Path(p).exists()]


def random_plate_number(rng: random.Random) -> str:
    region_len = 3 if rng.random() < 0.45 else 2
    if region_len == 3:
        first = rng.choice("127")
        region = first + "".join(rng.choice(DIGITS) for _ in range(2))
    else:
        region = rng.choice("123456789") + rng.choice(DIGITS)
    return (
        rng.choice(LETTERS)
        + "".join(rng.choice(DIGITS) for _ in range(3))
        + rng.choice(LETTERS)
        + rng.choice(LETTERS)
        + region
    )


@lru_cache(maxsize=256)
def _load_font(path: str, size: int):
    return ImageFont.truetype(path, size=size)

def _font(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    path = rng.choice(FONT_CANDIDATES) if FONT_CANDIDATES else None
    if path:
        return _load_font(path, size)
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], rng: random.Random, max_size: int) -> ImageFont.ImageFont:
    x1, y1, x2, y2 = box
    target_w = x2 - x1 - 3
    target_h = y2 - y1 - 3
    size = max_size
    while size >= 18:
        f = _font(size, rng)
        bb = draw.textbbox((0, 0), text, font=f, stroke_width=0)
        if bb[2] - bb[0] <= target_w and bb[3] - bb[1] <= target_h:
            return f
        size -= 2
    return _font(max(16, size), rng)


def _draw_char(draw: ImageDraw.ImageDraw, char: str, box: tuple[int, int, int, int], rng: random.Random, max_size: int, fill=(5, 5, 5)) -> None:
    x1, y1, x2, y2 = box
    f = _fit_text(draw, char, box, rng, max_size)
    bb = draw.textbbox((0, 0), char, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    jitter_x = rng.randint(-1, 1)
    jitter_y = rng.randint(-1, 1)
    x = x1 + (x2 - x1 - w) / 2 - bb[0] + jitter_x
    y = y1 + (y2 - y1 - h) / 2 - bb[1] + jitter_y
    draw.text((x, y), char, font=f, fill=fill)


def render_plate(plate_num: str, plate_type: str, rng: random.Random) -> Image.Image:
    if plate_type == "type1a":
        w, h = TYPE1A_SIZE
    else:
        w, h = TYPE1_SIZE

    if plate_type == "type1b":
        bg = (245 + rng.randint(-8, 5), 194 + rng.randint(-10, 10), 35 + rng.randint(-5, 8))
    else:
        v = 245 + rng.randint(-8, 8)
        bg = (v, v, v)

    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    border = 3 if w > 300 else 2
    d.rounded_rectangle((2, 2, w - 3, h - 3), radius=7, outline=(20, 20, 20), width=border)

    if rng.random() < 0.35:
        for _ in range(rng.randint(2, 7)):
            x = rng.randint(3, w - 4)
            y = rng.randint(3, h - 4)
            r = rng.randint(1, 4)
            col = rng.randint(150, 220)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(col, col, col))

    main = plate_num[:6]
    region = plate_num[6:]

    if plate_type == "type1a":
        for ch, box in zip(main[:4], TYPE1A_TOP):
            _draw_char(d, ch, box, rng, 70)
        for ch, box in zip(main[4:6], TYPE1A_BOTTOM_SERIES):
            _draw_char(d, ch, box, rng, 68)
        rslots = TYPE1A_REGION_3 if len(region) == 3 else TYPE1A_REGION_2
        for ch, box in zip(region, rslots):
            _draw_char(d, ch, box, rng, 52)
        rus_font = _font(15, rng)
        d.text((195, 149), "RUS", font=rus_font, fill=(15, 15, 15))
        d.rectangle((239, 153, 247, 157), fill=(255, 255, 255))
        d.rectangle((247, 153, 255, 157), fill=(30, 90, 180))
        d.rectangle((255, 153, 263, 157), fill=(190, 35, 35))
    else:
        for ch, box in zip(main, TYPE1_MAIN_SLOTS):
            _draw_char(d, ch, box, rng, 82)
        rslots = TYPE1_REGION_3 if len(region) == 3 else TYPE1_REGION_2
        for ch, box in zip(region, rslots):
            _draw_char(d, ch, box, rng, 63)
        rus_font = _font(17, rng)
        d.text((404, 78), "RUS", font=rus_font, fill=(15, 15, 15))
        d.rectangle((458, 86, 468, 90), fill=(255, 255, 255))
        d.rectangle((468, 86, 478, 90), fill=(40, 90, 180))
        d.rectangle((478, 86, 488, 90), fill=(190, 35, 35))

    if rng.random() < 0.45:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.45)))
    return im


def render_other(rng: random.Random) -> tuple[Image.Image, str]:
    mode = rng.choice(["motorcycle", "foreign", "ad"])
    if mode == "motorcycle":
        w, h = 210, 155
        bg = (240, 240, 240)
        text = f"{rng.randint(10,99)} {rng.choice(LETTERS)}{rng.randint(100,999)}"
    elif mode == "foreign":
        w, h = 500, 110
        bg = (240, 240, 240)
        text = f"{rng.choice('WQZ')}{rng.randint(10,99)}-{rng.choice('JFL')}{rng.randint(100,999)}"
    else:
        w, h = 430, 110
        bg = (230, 230, 245)
        text = rng.choice(["AUTO 24", "SALE 2026", "TEST CAR", "DEMO 777"])
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((2, 2, w - 3, h - 3), radius=8, outline=(30, 30, 30), width=3)
    f = _font(max(34, int(h * 0.52)), rng)
    bb = d.textbbox((0, 0), text, font=f)
    tx = max(5, (w - (bb[2] - bb[0])) // 2)
    ty = max(3, (h - (bb[3] - bb[1])) // 2 - bb[1])
    d.text((tx, ty), text, font=f, fill=(10, 10, 10))
    return im, text


def _random_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    c1 = np.array([rng.randint(30, 180), rng.randint(30, 180), rng.randint(30, 180)], dtype=np.float32)
    c2 = np.array([rng.randint(40, 210), rng.randint(40, 210), rng.randint(40, 210)], dtype=np.float32)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    bg = c1[None, None, :] * (1 - yy) + c2[None, None, :] * yy
    bg = np.repeat(bg, width, axis=1)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(2.0, 9.0), bg.shape)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    x1 = rng.randint(20, width // 5)
    x2 = rng.randint(width * 4 // 5, width - 20)
    y1 = rng.randint(height // 4, height // 2)
    y2 = rng.randint(height * 3 // 4, height - 15)
    car_col = tuple(int(v) for v in [rng.randint(20, 210), rng.randint(20, 210), rng.randint(20, 210)])
    cv2.rectangle(bg, (x1, y1), (x2, y2), car_col, -1)
    cv2.ellipse(bg, ((x1 + x2) // 2, y1 + 5), ((x2 - x1) // 3, (y2 - y1) // 2), 0, 180, 360, car_col, -1)
    for cx in (x1 + (x2 - x1) // 4, x1 + 3 * (x2 - x1) // 4):
        cv2.circle(bg, (cx, y2), max(8, (y2 - y1) // 8), (20, 20, 20), -1)
    return bg


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def paste_plate_scene(
    plate: Image.Image,
    rng: random.Random,
    canvas_size: tuple[int, int] = (640, 360),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cw, ch = canvas_size
    bg = _random_background(cw, ch, rng)
    plate_np = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)
    ph, pw = plate_np.shape[:2]

    target_w = rng.randint(max(90, cw // 5), min(cw // 2, 300))
    scale = target_w / pw
    target_h = max(35, int(ph * scale))
    plate_np = cv2.resize(plate_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ph, pw = plate_np.shape[:2]

    cx = rng.randint(pw // 2 + 15, cw - pw // 2 - 15)
    cy = rng.randint(ch // 2, ch - ph // 2 - 18)
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])

    skew_x = rng.uniform(-0.10, 0.10) * pw
    skew_y = rng.uniform(-0.15, 0.15) * ph
    dst = np.float32([
        [cx - pw / 2 + rng.uniform(-8, 8), cy - ph / 2 + skew_y],
        [cx + pw / 2 + rng.uniform(-8, 8), cy - ph / 2 - skew_y],
        [cx + pw / 2 + skew_x + rng.uniform(-8, 8), cy + ph / 2 + rng.uniform(-5, 5)],
        [cx - pw / 2 + skew_x + rng.uniform(-8, 8), cy + ph / 2 + rng.uniform(-5, 5)],
    ])
    dst[:, 0] = np.clip(dst[:, 0], 1, cw - 2)
    dst[:, 1] = np.clip(dst[:, 1], 1, ch - 2)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate_np, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (cw, ch))
    m3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    out = (warped.astype(np.float32) * m3 + bg.astype(np.float32) * (1 - m3)).astype(np.uint8)

    conditions: list[str] = ["day"]
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.4, 1.1))
        conditions.append("motion_blur")
    if rng.random() < 0.22:
        alpha = rng.uniform(0.45, 0.75)
        out = np.clip(out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        conditions = [c for c in conditions if c != "day"] + ["night"]
    if rng.random() < 0.18:
        overlay = out.copy()
        x = rng.randint(0, cw - 1)
        cv2.line(overlay, (x, 0), (min(cw - 1, x + rng.randint(50, 180)), ch - 1), (255, 255, 255), rng.randint(3, 12))
        out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)
        conditions.append("glare")
    if abs(skew_x) > 0.035 * pw or abs(skew_y) > 0.04 * ph:
        conditions.append("angle")
    if rng.random() < 0.18:
        for _ in range(rng.randint(4, 12)):
            px = int(rng.uniform(dst[:, 0].min(), dst[:, 0].max()))
            py = int(rng.uniform(dst[:, 1].min(), dst[:, 1].max()))
            cv2.circle(out, (px, py), rng.randint(1, 4), (80, 75, 65), -1)
        conditions.append("dirt")

    return out, _order_quad(dst), conditions
