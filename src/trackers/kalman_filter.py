# src/trackers/kalman_filter.py
import cv2
import numpy as np

def create_kalman_instance():
    # 4 state: x, y, vx, vy ; 2 measurements: x,y
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.statePost = np.zeros((4,1), dtype=np.float32)
    return kf
