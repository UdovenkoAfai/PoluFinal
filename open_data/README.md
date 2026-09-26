# Open-data curation workflow

This folder is for legally sourced **real** image candidates. It is intentionally separated from the official `dataset/` until a human checks the plate class and transcription.

## Why review is mandatory

A Commons category or a model prediction is only a search hint, not ground truth. A kei/JDM car does not necessarily have a `type1a` plate; a taxi photo does not necessarily show a readable `type1b` plate. Therefore the downloader only fills a review queue and never silently writes internet images into `dataset/meta.csv`.

## Review

For each suitable image, fill `review_status=approved`, `plate_num`, `plate_type`, `bbox`, `quad`, `is_vehicle`, and optional `conditions`. Also visually check that visible faces are blurred/covered.

Then run:

```bash
python tools/promote_reviewed.py
python validate_local.py --dataset ./dataset
```
