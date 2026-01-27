@echo off
REM LiverSegNet GPU-Optimized Training Launcher
REM Fast iteration mode with automatic GPU optimization

setlocal enabledelayedexpansion

echo.
echo ========================================
echo  LiverSegNet - GPU Optimized Training
echo ========================================
echo.

REM Check if venv exists
if not exist "venv_cuda" (
    echo ❌ Error: venv_cuda not found
    echo Please run the main setup first
    pause
    exit /b 1
)

REM Check NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Warning: NVIDIA GPU not detected
    echo Continuing with CPU (slower)
    echo.
) else (
    echo ✅ NVIDIA GPU detected
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo.
)

REM Run training
echo 🚀 Starting GPU-optimized training...
echo    - Auto-detects optimal batch size
echo    - Mixed precision (FP16) enabled
echo    - cuDNN benchmark enabled
echo    - Fast iteration (30 epochs per model)
echo.

venv_cuda\Scripts\python.exe train_gpu_optimized.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Training completed successfully!
    echo    Check 'training_results/' for detailed metrics
) else (
    echo.
    echo ❌ Training failed
)

pause
