# src/utils/config.py
# Configuration shared across the project

# PID defaults (kept from previous)
PID_KP = 0.05
PID_KI = 0.0
PID_KD = 0.01

# Multi-camera configuration
# Edit 'source' values to your phone URLs or webcam index.
# If your phone URL changes between runs (dynamic), leave it as placeholder and update before running.
CAMERAS = [
    {"id": 0, "source": "http://192.168.91.150:8080/video", "homography": "cal/cam0_H.npy"},
    {"id": 1, "source": "http://192.168.1.102:8080/video", "homography": "cal/cam1_H.npy"},
    {"id": 2, "source": 0, "homography": "cal/cam2_H.npy"},  # laptop webcam index 0
]

# Fusion mode: 'homography' requires cal/*.npy homography files to exist
FUSION_MODE = "homography"

# DeepSort / tracker defaults (kept for compatibility)
DEEPSORT_MAX_AGE = 30
DEEPSORT_IOU_THRESHOLD = 0.3

# Output fused log path
FUSED_LOG_PATH = "src/logs/fused_log.csv"
