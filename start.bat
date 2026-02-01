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
echo 📁 Professional Directories:
echo    • ui_launcher.py        - UI selection interface
echo    • ui/app_v2.py          - Standard enhanced UI
echo    • ui/app_professional_medical.py - Professional medical UI
echo    • ui/app_advanced_surgical.py    - Advanced research UI
echo    • critical_frames/     - Test frames with liver + tools
echo    • results/            - Auto-saved professional analyses
echo.
echo 🚨 Medical Alert System:
echo    • Critical occlusion detection
echo    • Real-time performance monitoring
echo    • Professional medical metrics
echo    • Advanced analytics dashboard
echo.
echo ========================================
echo 💡 PRO TIP: Use the UI Launcher (http://localhost:8500) to choose
echo    the best interface for your specific medical use case!
echo.
pause
