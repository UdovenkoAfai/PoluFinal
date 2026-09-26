@echo off
setlocal
cd /d "%~dp0"
python tools\promote_reviewed.py
if errorlevel 1 (
  echo.
  echo Some approved rows are incomplete or invalid. Fix open_data\review_queue.csv and run again.
  pause
  exit /b 1
)
python validate_local.py --dataset .\dataset
echo.
echo Done. Check warnings above and SUBMISSION_CHECKLIST.md.
pause
