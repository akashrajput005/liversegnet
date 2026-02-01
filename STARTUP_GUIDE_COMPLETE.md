# 🚀 LiverSegNet AI - Complete Startup Guide

## 🎯 **Quick Start**

### **Option 1: Automatic Start (Recommended)**
```bash
# Double-click this file or run:
start.bat
```

### **Option 2: Manual Start**
```bash
# Interactive menu with options:
run_ui_manual.bat
```

### **Option 3: Command Line**
```bash
# Terminal 1 - Start API Server:
python ui/app_api_v2.py

# Terminal 2 - Start UI Dashboard:
streamlit run ui/app_v2.py --server.port 8501
```

## 🌐 **Access Points**

- **Main Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## ✨ **New Features - Auto-Save System**

### **🔄 Automatic Result Saving**
Every segmentation analysis automatically saves:
- **JSON Metadata**: Complete analysis with timestamps
- **Overlay Images**: Visual segmentation results
- **Mask Images**: Raw segmentation masks
- **Text Summaries**: Human-readable reports

### **📁 File Structure**
```
liversegnet/
├── start.bat              # Automatic startup script
├── run_ui_manual.bat      # Manual startup with options
├── ui/
│   ├── app_v2.py          # Enhanced UI with auto-save
│   └── app_api_v2.py      # API backend
├── critical_frames/       # Test frames (liver + tools)
├── results/              # Auto-saved analyses (created automatically)
└── uploads/              # User uploaded images (created automatically)
```

## 🧪 **Quick Test (5 Minutes)**

1. **Start the application**: Double-click `start.bat`
2. **Open browser**: Go to http://localhost:8501
3. **Navigate**: Click "📸 Image Segmentation" tab
4. **Upload test frame**: Choose `critical_frames/frame_00_original.png`
5. **Run analysis**: Click "🚀 Start Segmentation"
6. **View results**: Check "💾 Saved Results" tab
7. **Verify auto-save**: Look for success message and new files in `results/`

## 🎛️ **UI Features**

### **4 Main Tabs:**
1. **📸 Image Segmentation** - Upload and analyze images
2. **🎥 Video Processing** - Process surgical videos
3. **📈 Model Analytics** - View model performance
4. **💾 Saved Results** - Browse auto-saved analyses

### **Model Options:**
- **DeepLabV3+** (Advanced) - Best accuracy, slower
- **U-Net** (Fast) - Quick processing, good accuracy
- **Stage 1** (Liver-only) - Specialized anatomy detection
- **Ensemble Mode** - Combine multiple models

## 📊 **Auto-Save Details**

### **What Gets Saved:**
Every analysis creates 4 timestamped files:
```
results/
├── frame_00_original_20260201_143022_analysis.json    # Complete metadata
├── frame_00_original_20260201_143022_overlay.png      # Visual result
├── frame_00_original_20260201_143022_mask.png         # Segmentation mask
└── frame_00_original_20260201_143022_summary.txt       # Human-readable report
```

### **JSON Analysis Content:**
```json
{
  "timestamp": "20260201_143022",
  "original_filename": "frame_00_original.png",
  "model_used": "deeplabv3plus",
  "mode": "single_model",
  "inference_time": 0.366,
  "liver_detected": true,
  "instrument_detected": true,
  "occlusion_percent": 15.2,
  "distance_px": 45,
  "metrics": {
    "liver_area_percent": 19.6,
    "instrument_area_percent": 52.9,
    "liver_regions": 1,
    "instrument_regions": 2
  }
}
```

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Port already in use**: The script automatically stops existing processes
2. **API not responding**: Wait 10-15 seconds after starting API before UI
3. **No results saved**: Check that you're using the UI (not direct API calls)
4. **Missing frames**: Run `python extract_critical_frames.py` to recreate test frames

### **Manual Cleanup:**
```bash
# Clear all results
rmdir /s results

# Restart services
taskkill /f /im python.exe
start.bat
```

## 📈 **Performance**

- **Inference Time**: 0.3-0.4 seconds per image
- **Models Loaded**: 3 (U-Net, DeepLabV3+, Stage 1)
- **Auto-Save Time**: < 1 second additional
- **Memory Usage**: ~2-4GB GPU VRAM recommended

## 🎯 **Success Indicators**

✅ **Working Setup:**
- API server starts on port 8000
- UI dashboard starts on port 8501
- All 3 models loaded successfully
- Critical frames available for testing
- Auto-save creates files in `results/` directory

✅ **Test Successful:**
- Can upload and analyze images
- Results display in UI
- Success message appears
- Files created in `results/` directory
- "Saved Results" tab shows analyses

## 🚀 **Ready to Use!**

The LiverSegNet AI system is now fully operational with:
- ✅ **Automatic result saving**
- ✅ **Enhanced UI with 4 tabs**
- ✅ **Critical test frames ready**
- ✅ **Complete startup automation**
- ✅ **Comprehensive error handling**

**Start using it now by running `start.bat`!** 🎉
