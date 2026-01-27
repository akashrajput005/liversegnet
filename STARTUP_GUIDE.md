# LiverSegNet - Quick Startup Guide

## Running the Application

### Terminal 1: Start the API Backend
```powershell
cd C:\Users\Public\liversegnet
venv_cuda\Scripts\python.exe ui/app_api.py
```

### Terminal 2: Start the Streamlit Dashboard
```powershell
cd C:\Users\Public\liversegnet
venv_cuda\Scripts\python.exe -m streamlit run ui/app.py
```

### Access the Application
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## Troubleshooting

### If Port 8000 is Already in Use
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### If Port 8501 is Already in Use
```powershell
Get-Process -Name streamlit | Stop-Process -Force
```

### Restart Everything (Full Reset)
```powershell
# Stop all processes
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force

# Start API
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\Public\liversegnet; venv_cuda\Scripts\python.exe ui/app_api.py"

# Wait 3 seconds
Start-Sleep -Seconds 3

# Start UI
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\Public\liversegnet; venv_cuda\Scripts\python.exe -m streamlit run ui/app.py"
```

---

## Test Files Available
- `test_frame.png` - Original test image
- `surg_interaction_1.png`, `surg_interaction_2.png`, `surg_interaction_3.png` - Complex surgical scenes
- `test_surgical_clip.avi` - 30-frame surgical video sequence

---

## System Architecture
- **Backend**: FastAPI (Port 8000) - Handles AI inference with Triple-Head Ensemble
- **Frontend**: Streamlit (Port 8501) - Interactive dashboard
- **Models**: U-Net (Baseline) + DeepLabV3+ (Advanced) + Stage 1 Anatomical Anchor
