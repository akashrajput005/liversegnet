@echo off
echo ========================================
echo  LiverSegNet AI - Professional Suite
echo ========================================
echo.

cd /d "C:\Users\Public\liversegnet"

echo [1/4] Stopping any existing processes...
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" 2>nul
powershell -Command "Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force" 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Starting API Backend (Port 8000)...
start "LiverSegNet API" cmd /k ".venv\Scripts\python.exe ui\app_api_v2.py"
timeout /t 3 /nobreak >nul

echo [3/4] Starting Unified Medical UI (Port 8500)...
start "LiverSegNet Medical" cmd /k ".venv\Scripts\python.exe -m streamlit run ui\app_medical_unified.py --server.port 8500"

echo [4/4] Waiting for services to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo  � LiverSegNet Medical Suite Started!
echo ========================================
echo.
echo 🌐 Web Interface:
echo    http://localhost:8500
echo.
echo 🤖 API Server:
echo    http://localhost:8000
echo.
echo 💡 Features:
echo    • Patient View: Simple, easy-to-understand results
echo    • Professional View: Detailed metrics and analysis
echo    • Real-time GPU/CPU detection
echo    • Live performance monitoring
echo.
echo 📋 Troubleshooting:
echo    • If API error appears, check terminal windows
echo    • Make sure both API and UI are running
echo    • Use debug sections in UI for help
echo.
echo 🌐 Access Points:
echo    Medical Suite:   http://localhost:8500  ⭐ MAIN INTERFACE
echo    API Server:      http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo.
echo 🚀 Ready to use! Open your browser and go to:
echo    http://localhost:8500
echo.
echo 📁 Available UI Files:
echo    • ui/app_medical_unified.py        - MAIN INTERFACE (Patient + Professional)
echo    • ui/app_professional_medical.py   - Professional medical UI
echo    • ui/app_advanced_surgical.py      - Advanced research UI
echo    • ui/app_surgical_clean.py         - Clean surgical interface
echo    • ui/app_v2.py                     - Standard enhanced UI
echo    • critical_frames/                 - Test frames for AI testing
echo    • results/                         - Analysis results directory
echo.
echo 🚨 Current Features:
echo    • Unified Medical Suite with dual modes
echo    • Real-time AI detection and metrics
echo    • Image enhancement and preprocessing
echo    • Professional error handling
echo    • GPU/CPU detection and monitoring
echo.
echo ========================================
echo 💡 READY TO USE: Open http://localhost:8500
echo ========================================
pause
