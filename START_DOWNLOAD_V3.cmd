@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ============================================================
echo  OPEN DATA DOWNLOADER V3 - NO PYTHON REQUIRED
echo ============================================================
echo.
echo V3 fixes the Windows PowerShell encoding/parser error.
echo The PowerShell source is ASCII-only and also has a UTF-8 BOM.
echo It also keeps the V2 fixes for 414 and 429 errors.
echo.

if not exist "tools\fetch_open_data_no_python_v3.ps1" (
  echo ERROR: tools\fetch_open_data_no_python_v3.ps1 is missing.
  echo Extract the whole ZIP before running this file.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\fetch_open_data_no_python_v3.ps1"
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo ============================================================
  echo  DOWNLOAD FAILED - exit code %ERR%
  echo ============================================================
  echo Send a screenshot of the last lines above.
) else (
  echo ============================================================
  echo  DOWNLOAD FINISHED
  echo ============================================================
  echo Candidates: open_data\candidates\
  echo Review CSV: open_data\review_queue.csv
)
echo.
pause
exit /b %ERR%
