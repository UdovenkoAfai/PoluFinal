# Third-party components

- OpenCV Russian plate Haar cascade (`models/haarcascade_russian_plate_number.xml`) is distributed with OpenCV. OpenCV is licensed under Apache License 2.0.
- Python dependencies are listed in `requirements.txt` and are not vendored into this archive.
- The OCR SVM weights in `models/char_svm.npz` were trained locally from generated glyphs by `train_ocr.py`; no external OCR API or proprietary model is used.
