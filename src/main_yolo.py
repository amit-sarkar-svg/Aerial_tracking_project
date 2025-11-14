# Updated main_yolo.py with Auto-Recenter + Auto-Relock System
# Fully merged with your previous YOLO + Kalman + PID logic

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import cv2
import numpy as np

from src.utils.camera_stream import get_stream
from src.trackers.kalman_filter import create_kalman
from src.trackers.pid_controller import PID
from src.utils.draw_utils import draw_bbox, draw_center
from src.utils.config import PID_KP, PID_KI, PID_KD
from src.detectors.yolo_detector import YOLODetector

# ----------------------------
# Helper: OpenCV tracker factory
# ----------------------------
def create_tracker_by_name(name='csrt'):
    name = name.lower()
    if name == 'csrt' and hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if name == 'kcf' and hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if name == 'medianflow' and hasattr(cv2, 'TrackerMedianFlow_create'):
        return cv2.TrackerMedianFlow_create()
    if name == 'mosse' and hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        return None

# ----------------------------
# Main System
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--class-id", type=int, default=None)
    ap.add_argument("--tracker", type=str, default="csrt")
    ap.add_argument("--detect-every", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    # YOLO detector
    yolo = YOLODetector(model_name=args.model, device=args.device, conf_threshold=args.conf)
    DETECT_EVERY = max(1, int(args.detect_every))

    tracker = None
    bbox = None
    frames_since_detect = DETECT_EVERY

    # Video input
    src = args.source
    if src.isdigit(): src = int(src)
    cap = get_stream(src)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read source.")
        return

    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    # Kalman and PID
    kf = create_kalman()
    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)

    # Auto-Recenter Variables
    lost_counter = 0
    AUTO_LOST_FRAMES = 8
    search_x, search_y = cx_img, cy_img
    forces_redetect = False

    frame_idx = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detected_bbox = None

        # ----------------------------
        # 1. DETECTION (YOLO)
        # ----------------------------
        if tracker is None or frames_since_detect >= DETECT_EVERY or forces_redetect:
            dets = yolo.detect(frame, classes=None if args.class_id is None else [args.class_id])
            if dets:
                best = max(dets, key=lambda d: d['conf'])
                detected_bbox = best['bbox']
                tracker = create_tracker_by_name(args.tracker)

                if tracker is not None:
                    try:
                        tracker.init(frame, tuple(detected_bbox))
                    except:
                        tracker = None

                bbox = detected_bbox
                frames_since_detect = 0
                lost_counter = 0
                forces_redetect = False
                search_x, search_y = cx_img, cy_img  # reset search point
            else:
                frames_since_detect += 1

        # ----------------------------
        # 2. TRACKER UPDATE
        # ----------------------------
        else:
            if tracker is not None:
                ok, tracked_box = tracker.update(frame)
                if ok:
                    x, y, w_box, h_box = [int(v) for v in tracked_box]
                    bbox = (x, y, w_box, h_box)
                    detected_bbox = bbox
                    frames_since_detect += 1
                else:
                    tracker = None
                    bbox = None
                    frames_since_detect = DETECT_EVERY

        # ----------------------------
        # 3. TRACK LOST HANDLING
        # ----------------------------
        if detected_bbox is None:
            lost_counter += 1
        else:
            lost_counter = 0

        is_lost = lost_counter > AUTO_LOST_FRAMES

        # ----------------------------
        # 4. KALMAN UPDATE/PREDICT
        # ----------------------------
        if detected_bbox is not None:
            x, y, w_box, h_box = detected_bbox
            cx = x + w_box/2
            cy = y + h_box/2
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        pred = kf.predict()
        px, py = int(pred[0][0]), int(pred[1][0])

        # ----------------------------
        # 5. AUTO-RECENTER + AUTO-RELOCK
        # ----------------------------
        if is_lost:
            vx = float(kf.statePost[2])
            vy = float(kf.statePost[3])
            mag = max(1, abs(vx) + abs(vy))
            vx, vy = vx/mag, vy/mag

            search_x += int(vx * 25)
            search_y += int(vy * 25)
            search_x = max(0, min(w - 1, search_x))
            search_y = max(0, min(h - 1, search_y))

            cv2.circle(frame, (search_x, search_y), 8, (0, 255, 255), -1)
            forces_redetect = True

        # ----------------------------
        # 6. DRAW VISUALS
        # ----------------------------
        if bbox:
            draw_bbox(frame, bbox)
            if detected_bbox is not None:
                draw_center(frame, int(cx), int(cy), (0,255,0))

        draw_center(frame, px, py, (255,0,0))   # Kalman predicted
        draw_center(frame, cx_img, cy_img, (0,0,255))  # Frame center

        # ----------------------------
        # 7. PID CONTROL
        # ----------------------------
        err_x = px - cx_img
        err_y = py - cy_img
        ctrl_x = pid_x.compute(err_x)
        ctrl_y = pid_y.compute(err_y)

        cv2.putText(frame, f"ctrl_x={ctrl_x:.2f} ctrl_y={ctrl_y:.2f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Lost={is_lost}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # FPS
        if frame_idx % 15 == 0:
            now = time.time()
            fps = 15 / (now - fps_time)
            fps_time = now
            cv2.putText(frame, f"FPS:{fps:.1f}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("YOLO-Kalman-PID + AutoRelock", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            frames_since_detect = DETECT_EVERY

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()