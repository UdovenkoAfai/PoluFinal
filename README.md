# IMPORTANT - easiest Windows start

If Python is not installed, do **not** use the old fetch BAT. Double-click `1_DOWNLOAD_PHOTOS_NO_PYTHON.cmd`. It downloads open-data candidates using Windows PowerShell only. See `START_HERE.txt`.

# Полуфинал - распознавание нестандартных ГРЗ

Готовый офлайн baseline по заданию: поиск номерного знака, классификация `type1` / `type1a` / `type1b`, OCR символов, отбрасывание неподходящих кандидатов и формирование CSV.

## Быстрый запуск

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py --input ./sample_input --output result.csv
```

Точная команда для проверки на каталоге изображений:

```bash
python run.py --input /path/to/images --output result.csv
```

Выход: UTF-8 CSV с разделителем `;` и колонками:

```text
image;plate_num;plate_type;confidence
```

Если на изображении не найден надёжный целевой знак, строка не создаётся. Это допустимо для нецелевых знаков по условию задания.

## Что внутри

- `run.py` - основной запуск;
- `app/` - детектор, геометрическое выравнивание, определение типа и OCR;
- `models/haarcascade_russian_plate_number.xml` - офлайн-детектор OpenCV для обычных номеров как один из сигналов;
- `models/char_svm.npz` - обученный HOG+LinearSVM классификатор символов;
- `train_ocr.py` - воспроизводимое обучение OCR-классификатора;
- `generator/generate_dataset.py` - генератор синтетики с фиксируемым seed;
- `dataset/` - данные в требуемой структуре;
- `validate_local.py` - локальная проверка структуры и формата;
- `docs/Пояснительная_записка.md` - краткое описание архитектуры, данных, скорости и ограничений.

## Генерация синтетики

```bash
python generator/generate_dataset.py --out ./dataset --count 5000 --seed 20260926 --clean
python validate_local.py --dataset ./dataset
```

Генератор создаёт `type1`, `type1a`, `type1b` и негативные `other`, добавляет перспективу, ночь, размытие, блики и загрязнение, формирует `meta.csv`, `labels/*.txt` и воспроизводится одним seed.

## Легальный сбор реальных изображений из открытого интернета

В проект добавлен консервативный сборщик Wikimedia Commons. Он не берёт «просто картинки из Google»: для каждого файла проверяется собственная лицензия, сохраняются источник/автор/хэш, исключаются неизвестные лицензии и CC BY-SA, выполняется первый проход размытия лиц и удаление точных дублей.

На Windows проще всего запустить `START_DOWNLOAD_V3.cmd`. В терминале эквивалентная команда:

```bash
python tools/fetch_open_data.py --max-type1a 180 --max-type1b 350 --max-other 80 --max-type1 100
```

Кандидаты появятся в `open_data/candidates/`, а вся информация для проверки - в `open_data/review_queue.csv`. Поле `class_hint` и предсказания baseline - **только подсказки**, а не эталонная разметка. Перед переносом в официальный датасет нужно визуально подтвердить номер, тип, bbox/quad и отсутствие неразмытых лиц.

После проверки поставьте `review_status=approved` только подходящим строкам. На Windows можно запустить `PROMOTE_REVIEWED_WINDOWS.bat`, либо выполнить:

```bash
python tools/promote_reviewed.py
python validate_local.py --dataset ./dataset
```

## Переобучение OCR

```bash
python train_ocr.py --samples-per-char 220
```

После обучения вес сохраняется в `models/char_svm.npz`.

## Скорость

На машине сборки baseline на тестовых синтетических изображениях показывал около 50-60 мс/изображение. Это НЕ официальный замер на референсном Intel Core i5-7600 / GTX 1050 Ti и его нужно перепроверить перед сдачей. Решение CPU-only и не обращается к интернету или внешним API.

## Важное ограничение перед официальной сдачей

Синтетическая часть подготовлена, а для реальной части есть автоматизированный легальный сборщик открытых источников. Но интернет-кандидаты становятся официальными данными только после визуальной проверки и команды `tools/promote_reviewed.py`: категория автомобиля или результат модели не доказывают тип/текст номера.

Отладочный набор организаторов нельзя добавлять в обучающий датасет.
