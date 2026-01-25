@echo off
echo ========================================
echo  LiverSegNet - Starting Application
echo ========================================
echo.

cd /d "c:\Users\akash\OneDrive\Desktop\LiverSegNet"

echo [1/3] Stopping any existing processes...
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" 2>nul
powershell -Command "Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force" 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Starting API Backend (Port 8000)...
start "LiverSegNet API" cmd /k "venv_cuda\Scripts\python.exe ui/app_api.py"
timeout /t 3 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard (Port 8501)...
start "LiverSegNet UI" cmd /k "venv_cuda\Scripts\python.exe -m streamlit run ui/app.py"

echo.
echo ========================================
echo  Application Started!
echo ========================================
echo  Dashboard: http://localhost:8501
echo  API Docs:  http://localhost:8000/docs
echo ========================================
echo.
pause
