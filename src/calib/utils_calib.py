# src/calib/utils_calib.py
# Utility helpers for the calibration pipeline

import cv2
import numpy as np
import json
from pathlib import Path

def save_json(path, obj):
    """
    Save JSON with conversion of numpy types to lists/floats.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        return o

    def rec(o):
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rec(v) for v in o]
        return convert(o)

    with open(path, "w") as f:
        json.dump(rec(obj), f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_checkerboard_corners(gray, pattern_size):
    """
    Find and refine checkerboard inner corners.
    pattern_size = (cols, rows) where cols = inner columns, rows = inner rows.
    Returns refined corners (Nx1x2) or None.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    corners2 = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners2

def build_object_points(pattern_size, square_size_m):
    """
    Build object points for the checkerboard in world coords (Z=0).
    pattern_size = (cols, rows) inner corners.
    square_size_m in meters.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    objp *= square_size_m
    return objp
