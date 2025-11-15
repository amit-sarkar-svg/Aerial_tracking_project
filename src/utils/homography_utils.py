# src/utils/homography_utils.py
import numpy as np
import cv2

def load_homography(path):
    """
    Load homography matrix (3x3) saved as numpy .npy file.
    Returns None if path is None or file not found.
    """
    if path is None:
        return None
    try:
        H = np.load(path)
        if H.shape == (3,3):
            return H.astype(np.float64)
    except Exception:
        return None
    return None

def transform_bbox_to_world(bbox, H):
    """
    bbox: (x, y, w, h) in image pixels
    H: 3x3 homography mapping image -> world plane (units are user-defined, e.g., meters)
    Returns: world_bbox: (wx_min, wy_min, wx_max, wy_max) rectangle covering transformed corners
    and center (wx_c, wy_c)
    """
    if H is None:
        return None
    x, y, w, h = bbox
    pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)
    # convert to homogeneous and apply H
    pts_h = cv2.perspectiveTransform(pts.reshape(-1,1,2), H)  # returns shape (4,1,2)
    pts_w = pts_h.reshape(-1,2)
    wx_min = float(np.min(pts_w[:,0])); wy_min = float(np.min(pts_w[:,1]))
    wx_max = float(np.max(pts_w[:,0])); wy_max = float(np.max(pts_w[:,1]))
    cx = float((wx_min + wx_max) / 2.0)
    cy = float((wy_min + wy_max) / 2.0)
    return (wx_min, wy_min, wx_max, wy_max, cx, cy)

def world_bbox_to_tuple(world_bbox):
    # helper to convert to a simple dict
    wx_min, wy_min, wx_max, wy_max, cx, cy = world_bbox
    return {"x1": wx_min, "y1": wy_min, "x2": wx_max, "y2": wy_max, "cx": cx, "cy": cy}
