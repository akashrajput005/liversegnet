@echo off
setlocal enabledelayedexpansion

:: ========================================
::  LiverSegNet AI - Unified Launcher
:: ========================================

:menu
cls
echo ========================================
echo        LIVERSEGNET AI SYSTEM
echo ========================================
echo.
echo  [1] 🚀 Launch Medical UI (Surgeon + Patient View)
echo  [2] 🧠 Start Model Training (Liver + Instruments)
echo  [3] 🧪 Run Perception Test (Single Frame)
echo  [4] 🧹 Cleanup Results & Temp Files
echo  [5] ❌ Exit
echo.
echo ========================================
set /p choice="Choose an option [1-5]: "

if "%choice%"=="1" goto launch_ui
if "%choice%"=="2" goto start_training
if "%choice%"=="3" goto run_test
if "%choice%"=="4" goto cleanup
if "%choice%"=="5" goto end

goto menu

:launch_ui
echo.
echo [INFO] Starting LiverSegNet Medical UI...
echo [INFO] Close this window or press Ctrl+C to stop.
echo.
:: Run directly since no separate API server is needed with current app.py
python -m streamlit run app.py --server.port 8501
pause
goto menu

:start_training
echo.
echo [INFO] Starting Model Training Module...
echo [INFO] This requires a configured dataset in configs/config.yaml
echo.
python src/train.py
pause
goto menu

:run_test
echo.
echo [INFO] Running AI Perception Test...
python test_perception.py
echo.
echo [INFO] Check test_output/ directory for results.
pause
goto menu

:cleanup
echo.
echo [INFO] Cleaning up results and temporary files...
if exist "results" (
    echo [INFO] Clearing results directory...
    del /q results\* 2>nul
)
if exist "test_output" (
    echo [INFO] Clearing test_output directory...
    del /q test_output\* 2>nul
)
if exist "uploads" (
    echo [INFO] Clearing uploads directory...
    del /q uploads\* 2>nul
)
echo [OK] Cleanup complete.
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
