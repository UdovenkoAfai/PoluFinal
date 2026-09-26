OPEN DATA DOWNLOADER V3

1. Extract this ZIP to a simple local folder, for example:
   C:\plate_project_v3\

2. Run:
   START_DOWNLOAD_V3.cmd

3. Do not run old V1/V2 downloader files.

V3 specifically fixes the Windows PowerShell encoding/parser error that showed text like "Р..." and "ExpectedValueExpression".
The script is ASCII-only and includes a UTF-8 BOM for compatibility with Windows PowerShell 5.1.

If Wikimedia reports HTTP 429, leave the window open. The script waits and retries automatically.
Progress is saved to open_data\review_queue.csv and later runs resume from saved files.
