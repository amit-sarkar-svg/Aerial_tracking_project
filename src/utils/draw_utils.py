import cv2

def draw_bbox(frame, bbox):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

def draw_center(frame, x, y, color=(255,0,0)):
    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
