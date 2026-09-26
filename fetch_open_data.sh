#!/bin/sh
set -eu
python tools/fetch_open_data.py --max-type1a 180 --max-type1b 350 --max-other 80 --max-type1 100
printf '\nReview open_data/review_queue.csv; set review_status=approved only for checked rows.\n'
printf 'Then run: python tools/promote_reviewed.py\n'
