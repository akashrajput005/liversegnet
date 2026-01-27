@echo off
echo ========================================
echo  LiverSegNet - Starting Application
echo ========================================
echo.

cd /d "C:\Users\Public\liversegnet"

echo [1/3] Stopping any existing processes...
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" 2>nul
powershell -Command "Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force" 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Starting API Backend (Port 8000)...
start "LiverSegNet API" cmd /k ".venv\Scripts\python.exe ui\app_api_v2.py"
timeout /t 3 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard (Port 8501)...
start "LiverSegNet UI" cmd /k ".venv\Scripts\python.exe -m streamlit run ui\app_v2.py --server.port 8501"

echo.
echo ========================================
echo  Application Started!
echo ========================================
echo  Dashboard: http://localhost:8501
echo  API Docs:  http://localhost:8000/docs
echo ========================================
echo.
pause
