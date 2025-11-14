# 🚁 Aerial Object Tracking System
### **YOLOv8 + OpenCV Tracker + Kalman Filter + PID Controller + Auto-Recenter & Auto-Relock**

A fully modular **real-time object tracking pipeline** designed for **aerial robotics**, **computer vision**, and **drone follow-me systems**.

This system combines:
- ✔ YOLOv8 Object Detection  
- ✔ OpenCV Trackers (CSRT / KCF / MOSSE)  
- ✔ Kalman Filter motion prediction  
- ✔ PID-based alignment control  
- ✔ **Auto-Recenter + Auto-Relock System** (drone-style recovery)

The system tracks any object **smoothly**, **intelligently**, and **recovers automatically** when tracking is lost.

---

# ✨ Features

### 🎯 **YOLOv8 Real-Time Object Detection**
High-speed detection of:
- People
- Vehicles
- Balls
- Any custom YOLO class

Supports **CPU** and **CUDA GPU**.

---

### 🎯 **OpenCV Trackers Between YOLO Frames**
Improves FPS while keeping accuracy.
- CSRT (accurate)
- KCF (fast)
- MOSSE (very fast)

---

### 🎯 **Kalman Filter Smoothing**
- Predicts object motion
- Removes jitter
- Works even when YOLO misses frames
- Provides velocity for Auto-Relock

---

### 🎯 **PID Controller**
Used for stable object-centering control:
- Horizontal movement
- Vertical movement

Perfect for:
- Drone gimbal
- Robot steering
- Simulation

---

### 🆕 **Auto-Recenter + Auto-Relock System**
This new recovery module ensures continuous tracking.

When the object is **lost**:
- Uses **Kalman-predicted motion direction**
- Moves a **search point** in that direction
- Forces YOLO to re-detect
- Automatically **re-acquires (relocks)** the target

Exactly like **DJI Follow-Me** drones.

---

# 🔍 Visual Meaning of Tracking Dots

| Color | Meaning | Source |
|-------|---------|--------|
| 🟩 Green | Real detection | YOLO / Tracker |
| 🔵 Blue | Kalman predicted center | Smoothed center |
| 🟡 Yellow | Auto-recenter search point | Recovery mode |
| 🔴 Red | Frame center | PID target |

---

# 🖼 Tracking + Auto-Relock Diagram

### **1. Normal Tracking**
```
+-----------------------------+
|             🔴             |
|                             |
|            🔵               |
|             🟩              |
+-----------------------------+
```

### **2. Lost Tracking → Auto-Recenter**
```
Last known direction → →

+-----------------------------+
|                  🟡        |
|         (no detection)     |
+-----------------------------+
```

### **3. YOLO Re-Detects → Relock**
```
+-----------------------------+
|              🔴             |
|              🔵             |
|              🟩             |
+-----------------------------+
```

---

# 📁 Project Structure
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
│   │   ├── draw_utils.py
│   │   └── config.py
│   ├── main_yolo.py
│   └── main.py
│
├── videos/
├── requirements.txt
├── README.md
└── venv/
```

---

# ⚙️ Installation

### **1. Clone the Repo**
```
git clone <your-repo-url>
cd aerial_tracking_project
```

### **2. Activate venv**
```
venv\Scripts\activate
```    

### **3. Install Dependencies**
```
pip install -r requirements.txt
```

---

# ▶️ Run the Tracker

### **Default Webcam**
```
python src/main_yolo.py --source 0
```

### **Track Only a Specific Class**
Example: person (class 0)
```
python src/main_yolo.py --class-id 0 --source 0
```

### **Use a Faster OpenCV Tracker**
```
python src/main_yolo.py --tracker mosse
```

### **Lower YOLO Frequency (improves FPS)**
```
python src/main_yolo.py --detect-every 20
```

---

# 🔧 Configuration
Modify `src/utils/config.py` to adjust:
- PID gains (KP, KI, KD)
- Detection thresholds
- Auto-Recenter parameters

---

# 🧠 How It Works

### **1. YOLO detects object (every N frames).**
### **2. OpenCV tracker follows in-between.**
### **3. Kalman filter predicts motion and smooths output.**
### **4. PID computes corrections to center the object.**
### **5. If object is lost → Auto-Recenter + Auto-Relock recovers it.**

---

# 🚀 Future Enhancements
- Multi-object tracking (DeepSORT / ByteTrack)
- Distance estimation (3D tracking)
- PX4 SITL drone control
- Real gimbal servo control
- ONNX/TensorRT acceleration
- GUI Panel (Tkinter / PyQt)

---

# 📜 License
MIT License

---

# 💬 Support
Need help or want to add new features?  
Feel free to ask anytime!

