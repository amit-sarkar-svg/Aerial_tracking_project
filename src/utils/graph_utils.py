# src/utils/graph_utils.py
import cv2
import numpy as np

def draw_distance_graph(frame, history, max_points=80):
    if len(history) < 2:
        return frame
    h, w = frame.shape[:2]
    graph_w = 220
    graph_h = 100
    x0 = w - graph_w - 10
    y0 = 10
    cv2.rectangle(frame, (x0,y0),(x0+graph_w,y0+graph_h),(30,30,30), -1)
    cv2.rectangle(frame, (x0,y0),(x0+graph_w,y0+graph_h),(200,200,200),1)
    hist = history[-max_points:]
    max_d = max(hist); min_d = min(hist)
    rng = max(0.01, max_d - min_d)
    pts = []
    for i,d in enumerate(hist):
        nx = x0 + int((i/ max_points) * graph_w)
        ny = y0 + graph_h - int(((d - min_d)/rng) * graph_h)
        pts.append((nx,ny))
    for i in range(1,len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0,255,0), 2)
    cv2.putText(frame, "Dist (m)", (x0+5, y0+12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return frame
