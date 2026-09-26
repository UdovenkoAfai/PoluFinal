#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = [
    "image", "plate_num", "plate_type", "bbox", "quad",
    "is_vehicle", "is_synthetic", "source", "license", "conditions"
]
TYPES = {"type1", "type1a", "type1b", "other"}
PLATE_RE = re.compile(r"^[ABEKMHOPCTYX][0-9#]{3}[ABEKMHOPCTYX#]{2}[0-9#]{2,3}$")
ALLOWED_LICENSES = {"CC BY 4.0", "CC0 1.0", "Public domain"}

def parse_ints(text: str, count: int, name: str) -> list[int]:
    try:
        vals = [int(v.strip()) for v in text.split(",")]
    except Exception as e:
        raise ValueError(f"{name} must contain integers") from e
    if len(vals) != count:
        raise ValueError(f"{name} must contain {count} integers")
    return vals

def load_meta(meta: Path) -> list[dict]:
    if not meta.exists():
        return []
    with meta.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))

def write_meta(meta: Path, rows: list[dict]) -> None:
    with meta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

def main() -> int:
    p = argparse.ArgumentParser(description="Promote manually reviewed open-data candidates into dataset/images/real and meta.csv")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--queue", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    queue = (args.queue or (root / "open_data" / "review_queue.csv")).resolve()
    if not queue.exists():
        raise SystemExit(f"queue not found: {queue}")

    dataset = root / "dataset"
    real_dir = dataset / "images" / "real"
    labels_dir = dataset / "labels"
    real_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta = dataset / "meta.csv"
    existing = load_meta(meta)
    existing_sha = set()
    existing_sources = set()
    for r in existing:
        if r.get("is_synthetic") == "0":
            existing_sources.add(r.get("source", ""))
            img = dataset / r.get("image", "")
            if img.exists():
                try:
                    existing_sha.add(hashlib.sha256(img.read_bytes()).hexdigest())
                except Exception:
                    pass

    with queue.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    skipped = 0
    problems = []
    new_rows = []
    for line_no, r in enumerate(rows, start=2):
        if r.get("review_status", "").strip().lower() != "approved":
            continue
        try:
            lic = r.get("license", "").strip()
            if lic not in ALLOWED_LICENSES:
                raise ValueError(f"license {lic!r} is not in the conservative allow-list")
            typ = r.get("plate_type", "").strip()
            if typ not in TYPES:
                raise ValueError(f"invalid plate_type {typ!r}")
            plate = r.get("plate_num", "").strip().upper()
            if typ == "other":
                if not plate:
                    plate = "########"
            elif not PLATE_RE.fullmatch(plate):
                raise ValueError(f"invalid plate_num {plate!r}")
            bbox = parse_ints(r.get("bbox", ""), 4, "bbox")
            quad = parse_ints(r.get("quad", ""), 8, "quad")
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError("bbox width/height must be positive")
            is_vehicle = r.get("is_vehicle", "").strip()
            if is_vehicle not in {"0", "1"}:
                raise ValueError("is_vehicle must be 0 or 1")
            src = (root / r.get("local_file", "")).resolve()
            if not src.exists() or root not in src.parents:
                raise ValueError(f"candidate file missing or outside project: {src}")
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            source_page = r.get("source_page", "").strip()
            if digest in existing_sha or (source_page and source_page in existing_sources):
                skipped += 1
                continue
            ext = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
            name = f"open_{digest[:16]}{ext}"
            dest = real_dir / name
            rel = dest.relative_to(dataset).as_posix()
            row = {
                "image": rel,
                "plate_num": plate,
                "plate_type": typ,
                "bbox": ",".join(str(x) for x in bbox),
                "quad": ",".join(str(x) for x in quad),
                "is_vehicle": is_vehicle,
                "is_synthetic": "0",
                "source": source_page or r.get("source_file_url", ""),
                "license": lic,
                "conditions": r.get("conditions", "").strip(),
            }
            new_rows.append((src, dest, row, digest))
        except Exception as e:
            problems.append(f"queue line {line_no}: {e}")

    if problems:
        for x in problems[:50]:
            print("ERROR:", x)
        print("Nothing was modified. Fix approved rows and rerun.")
        return 2

    if args.dry_run:
        print(f"DRY RUN: would promote {len(new_rows)} files, skip {skipped} duplicates")
        return 0

    promoted = 0
    for src, dest, row, digest in new_rows:
        shutil.copy2(src, dest)
        (labels_dir / (dest.stem + ".txt")).write_text(";".join(row[k] for k in FIELDNAMES) + "\n", encoding="utf-8")
        existing.append(row)
        existing_sha.add(digest)
        existing_sources.add(row["source"])
        promoted += 1

    write_meta(meta, existing)
    print(f"Promoted: {promoted}; duplicates skipped: {skipped}; meta rows now: {len(existing)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
