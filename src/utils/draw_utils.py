# src/utils/draw_utils.py
import cv2

_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
def _color_for_id(track_id):
    return _COLORS[track_id % len(_COLORS)]

def draw_bbox_id(frame, bbox, track_id, label=None, distance=None, velocity=None, angle=None):
    x,y,w,h = bbox
    color = _color_for_id(track_id)
    x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
    # compose text
    texts = []
    if label:
        texts.append(label)
    if distance is not None and distance != "":
        try:
            texts.append(f"d:{float(distance):.2f}m")
        except:
            texts.append(f"d:{distance}")
    if velocity is not None and velocity != "":
        try:
            texts.append(f"v:{float(velocity):.2f}m/s")
        except:
            texts.append(f"v:{velocity}")
    if angle is not None and angle != "":
        try:
            texts.append(f"{float(angle):.1f}°")
        except:
            texts.append(f"{angle}")
    text = " | ".join(texts)
    y_text = max(0, y1 - 8)
    cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

def draw_center(frame, x, y, color=(255,0,0), radius=4):
    cv2.circle(frame, (int(x),int(y)), radius, color, -1)
