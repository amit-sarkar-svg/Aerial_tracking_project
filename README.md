# 🚁 Aerial Object Tracking System (Multi-Object Version)
### **YOLOv8 + DeepSORT + Kalman + PID + Distance/Velocity/Angle Estimation + CSV Logging + Graphs**

A powerful **multi-object tracking system** designed for **aerial robotics**, **drone follow-me systems**, and **computer vision research**, now supporting:

- ✔ Multi-object tracking with **DeepSORT**
- ✔ YOLOv8 detection
- ✔ Kalman smoothing per object
- ✔ PID alignment per object
- ✔ Distance estimation (meters)
- ✔ Velocity estimation (m/s)
- ✔ Angle estimation (degrees)
- ✔ Per-object mini distance graph
- ✔ Per-object CSV logging
- ✔ Auto-recenter + re-detect (single-target mode)

---

# ✨ Features

### 🎯 **YOLOv8 Real-Time Multi-Class Detection**
Detects all COCO classes:
- People
- Cars
- Bikes
- Balls
- Animals
- Custom-trained models

---

### 🎯 **DeepSORT Multi-Object Tracking**
Each object gets:
- Unique **Track ID**
- Motion-based re-identification
- Stable tracking after occlusions
- Smooth motion via Kalman filter

---

### 🎯 **Distance, Velocity & Angle Estimation**
For every object:
- **Distance** (meters)
- **Velocity** (meters per sec)
- **Angle** relative to camera center

Formula:
```
Distance = (RealWidth * FocalLength) / PixelWidth
Angle = atan((cx - center_x) / focal_length)
```

---

### 🎯 **Right-Side Mini Graphs**
Each tracked object shows:
- Recent distance history
- Smooth trend line

---

### 🎯 **CSV Logging (Track-wise)**
Saved to:
```
src/logs/multi_log.csv
```
Columns:
```
timestamp, frame, track_id, class_id, class_name,
distance_m, velocity_m_s, angle_deg
```

---

### 🟡 Auto-Recenter + Auto-Relock (Single Object Mode)
When using `main_yolo.py`:
- Predicts where object moved
- Re-detects automatically
- Perfect for drone-style follow-me

---

# 📁 Project Structure
```
aerial_tracking_project/
│
├── src/
│   ├── main_yolo.py                 # single-object tracker
│   ├── main_yolo_multi.py           # multi-object tracker
│   ├── calibrate.py                 # focal calibration
│   │
│   ├── detectors/
│   │   └── yolo_detector.py
│   │
│   ├── trackers/
│   │   ├── kalman_filter.py
│   │   ├── pid_controller.py
│   │   └── deep_sort/
│   │       ├── deep_sort.py
│   │       ├── detection.py
│   │       ├── track.py
│   │       └── nn_matching.py
│   │
│   ├── utils/
│   │   ├── draw_utils.py
│   │   ├── graph_utils.py
│   │   ├── camera_stream.py
│   │   ├── object_sizes.py
│   │   └── config.py
│   │
│   └── logs/
│       └── multi_log.csv
│
├── videos/
├── requirements.txt
├── README.md
└── venv/
```

---

# ⚙️ Installation

### 1️⃣ Clone Repository
```
git clone <repo-url>
cd aerial_tracking_project
```

### 2️⃣ Activate Virtual Environment
```
venv\Scripts\activate
```

### 3️⃣ Install Requirements
```
pip install -r requirements.txt
```

---

# ▶️ Run Multi-Object Tracker

### Use webcam:
```
python src/main_yolo_multi.py --source 0
```

### Use video file:
```
python src/main_yolo_multi.py --source videos/test.mp4
```

### With calibrated focal length:
```
python src/main_yolo_multi.py --source 0 --focal 930
```

### Lower confidence threshold:
```
python src/main_yolo_multi.py --conf 0.25
```

---

# ▶️ Run Single-Object Tracker (Auto-Relock)
```
python src/main_yolo.py --source 0
```

Force detection every N frames:
```
python src/main_yolo.py --detect-every 5
```

Track a specific class:
```
python src/main_yolo.py --class-id 0
```

---

# 🎯 Focal Calibration
```
python src/calibrate.py
```
Steps:
1. Enter real object width (in meters)
2. Enter real distance (meters)
3. Draw bounding box
4. Script outputs focal length
5. Save & use in multi-object tracker

---

# 🖼 Visual Guide

| Symbol | Meaning |
|--------|---------|
| 🟩 | Real detection center |
| 🔵 | Kalman predicted center |
| 🎨 | Random color per track ID |
| 📈 | Mini graph (distance history) |
| 🔴 | Camera center |

---

# 🧠 System Pipeline
1. YOLO detects objects
2. DeepSORT assigns track IDs
3. Kalman filter smooths motion
4. PID aligns object center
5. Distance/Velocity/Angle computed
6. Distance graph generated
7. CSV logged per object

---

# 🚀 Future Add-ons
- Multi-camera fusion
- 3D triangulation
- Drone autopilot via MAVSDK
- Web dashboard
- TensorRT optimization
- ReID deep features for better DeepSORT

---

# 📜 License
MIT License

---

# 💬 Support
Need help with upgrades, enhancements, or debugging? I’m here to help!

