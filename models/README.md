# YOLO Pothole Detection Models Directory

This directory contains the YOLO object detection weights (`.pt` files) used by the pothole detection engine.

---

## 📍 Where to Place Your Custom Model

Place your custom trained YOLO weights file in this folder:

```
PotHole/
└── models/
    ├── pothole_yolo.pt     <-- RECOMMENDED: Place your trained weights here
    ├── best.pt             <-- Also automatically detected as fallback
    └── README.md
```

### Supported File Names (in order of priority):
1. `pothole_yolo.pt` (Default primary model path defined in `config/settings.py`)
2. `best.pt` (Common output name from Ultralytics YOLO training)
3. Any custom name specified via the `POTHOLE_MODEL_PATH` environment variable or command line argument `--model-path`.

---

## 🤖 Automatic Pretrained Weights Download

If no `.pt` model file is found in this directory when the system starts up, the built-in weight loader will **automatically download** a pre-trained, high-accuracy YOLOv8 pothole detection model (`peterhdd/pothole-detection-yolov8`) from Hugging Face into `models/pothole_yolo.pt`.

You can also manually trigger the download at any time:
```bash
python utils/download_weights.py
```

---

## 🔄 How to Replace or Update the Model

To replace the model with a new custom-trained checkpoint:
1. Copy your trained `.pt` file into this directory and rename it to `pothole_yolo.pt` (or update `config/settings.py`).
2. The detection engine will automatically load the new model without needing any code changes.
3. If the server is running, restart it or use the model reload trigger to activate the new weights.

---

## 🎯 Model Training Information
The model detects potholes on asphalt and concrete road surfaces.
- **Classes**: `pothole` (Class 0)
- **Framework**: Ultralytics YOLOv8 / YOLOv9 / YOLOv10 / YOLO11
- **Input Resolution**: 640x640 px (auto-letterboxed during inference)
