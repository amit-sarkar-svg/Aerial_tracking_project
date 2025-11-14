# config.py

# HSV color range for the target object
# (Example: red colored object — you must tune this for your object)
HSV_LOWER = (0, 120, 70)
HSV_UPPER = (10, 255, 255)

# Minimum contour area to consider a valid detection
MIN_AREA = 200

# PID gains (initial values)
PID_KP = 0.005
PID_KI = 0.0
PID_KD = 0.002
