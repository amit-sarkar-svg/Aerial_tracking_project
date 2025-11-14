import cv2
import numpy as np
from src.utils.config import HSV_LOWER, HSV_UPPER, MIN_AREA

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_LOWER)
    upper = np.array(HSV_UPPER)

    mask = cv2.inRange(hsv, lower, upper)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, mask

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h), mask
