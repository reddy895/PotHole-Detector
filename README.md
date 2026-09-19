# AI-Powered Pothole Detection System (Terminal Edition)

A high-performance, real-time computer vision application built with **Python**, **Ultralytics YOLO**, and **OpenCV** to detect and highlight road potholes from live webcams, images, and recorded road videos directly from your terminal.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*(If using an NVIDIA GPU with CUDA acceleration on Linux/Windows, ensure PyTorch with CUDA support is installed)*:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Run the Application

#### 📷 Webcam Mode (Real-Time Live Detection)
```bash
python main.py --source webcam
```
*Press **Q** or **ESC** in the preview window to stop.*

#### 🖼️ Image Mode (Single Image Detection)
```bash
python main.py --source image --input road.jpg
```
*Detects potholes in the road image, displays the OpenCV window, and automatically saves the annotated result to `outputs/`.*

#### 🎥 Video Mode (Recorded Video / Dashcam Detection)
```bash
python main.py --source video --input road.mp4
```
*Processes the video frame-by-frame, shows real-time detections, and saves the annotated video to `outputs/annotated_road.mp4`.*

---

## 📁 Project Structure

```
PotHole/
├── main.py              # Application CLI entrypoint (webcam, image, video dispatch)
├── detector.py          # Decoupled YOLO pothole inference engine & OpenCV visual rendering
├── config.py            # Centralized system configurations, model paths, and device selector
├── requirements.txt     # Python dependencies
├── README.md            # System documentation and execution guide
│
├── models/
│   └── pothole.pt       # Trained YOLO pothole weights
│
├── outputs/             # Output directory for annotated images and processed videos
│
├── utils/
│   ├── __init__.py
│   └── video_utils.py   # Stream I/O, terminal telemetry formatter, and file writer
│
├── road.jpg             # Pre-packaged sample road pothole image for instant testing
└── road.mp4             # Pre-packaged sample road video for instant testing
```

---

## 🎯 Model Placement

The detector automatically loads the trained YOLO model from:
```
models/pothole.pt
```

### To use your own custom-trained model:
1. Place your trained YOLO `.pt` file into the `models/` directory.
2. Ensure it is named `pothole.pt` (or specify `--model path/to/your_model.pt` when running).
3. The detector will automatically load and optimize the model for your GPU.

---

## ⚡ Hardware Acceleration (CUDA & CPU)

The system automatically detects whether an NVIDIA GPU with CUDA is available:
- **GPU (CUDA)**: Utilizes GPU tensor cores for high FPS real-time video and live webcam feeds.
- **CPU**: Gracefully falls back to CPU execution if no CUDA device is present.

To force a specific device, set the environment variable:
```bash
export POTHOLE_DEVICE=cpu    # Force CPU
export POTHOLE_DEVICE=cuda   # Force CUDA
```

---

## 🖥️ Terminal Output

While running, the system outputs structured telemetry directly to the console:

```text
========================================
      POTHOLE DETECTION SYSTEM
========================================
Model:  models/pothole.pt
Device: CUDA (NVIDIA GeForce RTX 2050)
Source: Webcam (Device 0)
Status: Running
========================================

[POTHOLE DETECTED]
Count: 2
Highest Confidence: 91.4%
FPS: 32
-------------------------

[NO POTHOLE]
FPS: 35
-------------------------
```

---

## ⚙️ Command-Line Options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--source` | `webcam`, `image`, `video` | **Required**. Selection mode. | None |
| `--input` | `path/to/file` | Path to image or video file. | None |
| `--model` | `path/to/model.pt` | Path to custom YOLO model weights. | `models/pothole.pt` |
| `--conf` | `float` (0.0 to 1.0) | Confidence threshold for detection. | `0.35` |
| `--cam-idx` | `int` | Camera index (for external or USB cameras). | `0` |
| `--no-view` | flag | Run in headless mode without opening GUI windows. | False |

---

## 🔮 Future Extensibility

The `PotholeDetector` class in `detector.py` is strictly decoupled from the UI and input handling. It includes an alert hook:

```python
detector.register_alert_handler(your_callback_function)
```

This allows you to attach future modules without touching the core detection engine:
- **WhatsApp Bot**: Send incident alerts and snapshots automatically.
- **GPS Telemetry**: Tag coordinates of detected potholes for municipal road maintenance.
- **Hazard Database**: Log pothole severity, size, and frequency to a database.
- **Multi-class Road Hazard Models**: Easily swap weights to detect cracks, debris, or speed bumps.
# PotHole-Detector
