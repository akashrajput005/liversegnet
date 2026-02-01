@echo off
echo 🚀 Starting Optimized U-Net Training - Best of Best Setup
echo ========================================================
echo.
echo This optimized training includes:
echo   • U-Net ResNet50 (upgraded from ResNet34)
echo   • 60 epochs (same as DeepLabV3+)
echo   • 5e-5 learning rate (same as DeepLabV3+)
echo   • ReduceLROnPlateau scheduler
echo   • Adaptive loss weighting
echo   • Enhanced gradient clipping
echo   • EMA for stable validation
echo   • Metric reset at epoch 30
echo.
echo Press any key to start...
pause > nul

python train_unet_optimized.py

echo.
echo Training complete! Check the logs for results.
pause
