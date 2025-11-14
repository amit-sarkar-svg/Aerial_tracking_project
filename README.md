# Aerial Tracking Project (YOLO + Kalman Filter + PID)

This project implements a **real‑time object tracking system** using:

- **YOLOv8** for object detection
- **OpenCV CSRT/MOSSE** for frame‑to‑frame tracking
- **Kalman Filter** for prediction & smoothing
- **PID Controller** for camera/drone alignment logic

It is designed for:  
✔ Aerial robotics  
✔ Autonomous tracking  
✔ Drone control research  
✔ Computer vision experiments

---

## 🚀 Features

### 🔹 YOLOv8 Object Detection
- Detects people, vehicles, balls, etc.
- Works in real‑time on CPU
- Supports class filtering (e.g., only track person)

### 🔹 Advanced Tracking Pipeline
- YOLO detects every N frames
- CSRT/MOSSE tracker handles interim frames
- Kalman Filter predicts object motion
- PID computes follow‑up control signals

### 🔹 Visual Indicators
- 🟩 **Green Dot** → Actual detection from YOLO/Tracker  
- 🔵 **Blue Dot** → Kalman predicted object position  
- 🔴 **Red Dot** → Frame center (target alignment point)

### 🔹 Fully Modular Code Structure
- `detectors/` → YOLO detectors  
- `trackers/` → Kalman + PID  
- `utils/` → helpers, drawing, streaming  
- `main_yolo.py` → Main tracking engine

---

## 📁 Project Structure

```
aerial_tracking_project/
│
├── src/
│   ├── detectors/
│   │   └── yolo_detector.py
│   ├── trackers/
│   │   ├── kalman_filter.py
│   │   └── pid_controller.py
│   ├── utils/
│   │   ├── camera_stream.py
│   │   ├── config.py
│   │   └── draw_utils.py
│   ├── main_yolo.py
│   └── main.py   (color-based tracker)
│
├── videos/        (sample test videos)
├── venv/          (virtual environment)
└── requirements.txt
```

---

## 🛠 Installation Guide

### 🔸 1. Install Python 3.10
YOLO & PyTorch require **Python 3.10**.  
Check version:
```
python --version
```

### 🔸 2. Create & Activate Virtual Environment
```
py -3.10 -m venv venv
venv\Scripts\activate
```

### 🔸 3. Install PyTorch 2.5.1 (CPU version)
```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

### 🔸 4. Install Other Dependencies
```
pip install -r requirements.txt
```

Where `requirements.txt` contains:
```
ultralytics==8.1.0
opencv-python==4.9.0.80
numpy==1.26.4
imutils==0.5.4
```

---

## ▶️ Run YOLO Tracking

### Run webcam tracking:
```
python src/main_yolo.py --source 0
```

### Run tracking on a video file:
```
python src/main_yolo.py --source videos/sample.mp4
```

### Track only a specific class (example: person → class 0)
```
python src/main_yolo.py --class-id 0
```

---

## ⚙️ How It Works

### 🔹 1. YOLO detects objects  
Runs every N frames (`--detect-every 10`).

### 🔹 2. CSRT/MOSSE tracks between detections  
Reduces compute load.

### 🔹 3. Kalman Filter predicts next motion  
Provides stable & smooth tracking.

### 🔹 4. PID Controller computes alignment offsets  
Used for drone/gimbal follow control.

---

## 🔍 Visual Meaning of Dots

| Color | Meaning | Source |
|-------|---------|--------|
| 🟩 Green | Actual detected position | YOLO/Tracker |
| 🔵 Blue | Predicted smoothed position | Kalman Filter |
| 🔴 Red | Camera center | PID target point |

---

## 🔧 Common Arguments

| Argument | Description |
|----------|-------------|
| `--source` | Webcam index or video file path |
| `--class-id` | Track specific object class |
| `--device` | `cpu` or `cuda` |
| `--detect-every` | YOLO detection interval |
| `--tracker` | `csrt`, `mosse`, `kcf` |

---

## 🛰 Future Improvements

- Multi‑object tracking with ID assignment  
- Integration with PX4 SITL for drone control  
- Depth estimation for distance measurement  
- Gimbal stabilization control  
- Faster ONNX or TensorRT YOLO models

---

## 🏁 Conclusion
This project provides a robust computer vision tracking pipeline that is suitable for:

- Drone object following  
- Aerial robotics research  
- Surveillance systems  
- AI‑based vision projects

If you want more enhancements, feel free to request advanced features! 🚀

