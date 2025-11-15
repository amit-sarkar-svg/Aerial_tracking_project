# src/utils/config.py

# PID defaults
PID_KP = 0.05
PID_KI = 0.0
PID_KD = 0.01

# Multi-camera configuration (list of cameras)
# Each camera: dict with keys:
#   id: integer id
#   source: camera source string or int (0, 1, "rtsp://...", or "videos/cam0.mp4")
#   homography: path to .npy file containing 3x3 homography (image -> world plane). If None, no homography used.
# Example:
CAMERAS = [
    {"id": 0, "source": 0, "homography": "cal/cam0_H.npy"},
    {"id": 1, "source": 1, "homography": "cal/cam1_H.npy"},
    # Add more cameras as needed...
]

# Fusion mode: "homography" to fuse via world-plane mapping;
# If homography files are missing or set to None for any camera, fusion will fall back to per-camera tracking only.
FUSION_MODE = "homography"  # or "independent"

# Global DeepSort parameters
DEEPSORT_MAX_AGE = 30
DEEPSORT_IOU_THRESHOLD = 0.3

# Output fused log
FUSED_LOG_PATH = "src/logs/fused_log.csv"
