@echo off
echo ========================================
echo  LiverSegNet AI - Manual UI Commands
echo ========================================
echo.

echo 🚀 Starting LiverSegNet AI with Auto-Save Feature
echo.

echo 📋 Available Commands:
echo.
echo 1️⃣ Start API Server:
echo    python ui/app_api_v2.py
echo.
echo 2️⃣ Start UI Dashboard:
echo    streamlit run ui/app_v2.py --server.port 8501
echo.
echo 3️⃣ Quick Test Auto-Save:
echo    python test_ui_with_frames.py
echo.
echo 4️⃣ Extract Critical Frames:
echo    python extract_critical_frames.py
echo.
echo 🌐 Access URLs:
echo    Dashboard: http://localhost:8501
echo    API Docs:  http://localhost:8000/docs
echo.
echo 📁 Important Directories:
echo    • critical_frames/ - Test frames with liver + tools
echo    • results/       - Auto-saved segmentation results
echo    • uploads/       - User uploaded images
echo.
echo ✨ New Features:
echo    • Automatic result saving to 'results/' folder
echo    • Enhanced UI with 4 tabs (including Saved Results)
echo    • Complete metadata preservation with timestamps
echo.
echo 🧪 Quick Test Steps:
echo    1. Run: python ui/app_api_v2.py
echo    2. Run: streamlit run ui/app_v2.py --server.port 8501
echo    3. Open: http://localhost:8501
echo    4. Upload: critical_frames/frame_00_original.png
echo    5. Click: "🚀 Start Segmentation"
echo    6. Check: "💾 Saved Results" tab
echo.
echo ========================================
echo.

set /p choice="Choose an option (1-4) or press Enter to exit: "

if "%choice%"=="1" (
    echo Starting API Server...
    python ui/app_api_v2.py
) else if "%choice%"=="2" (
    echo Starting UI Dashboard...
    streamlit run ui/app_v2.py --server.port 8501
) else if "%choice%"=="3" (
    echo Testing UI with Critical Frames...
    python test_ui_with_frames.py
) else if "%choice%"=="4" (
    echo Extracting Critical Frames...
    python extract_critical_frames.py
) else (
    echo Exiting...
)

pause
