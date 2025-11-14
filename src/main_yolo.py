# src/main_yolo.py
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
    # Compatibility across OpenCV versions
    if name == 'csrt' and hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if name == 'kcf' and hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if name == 'medianflow' and hasattr(cv2, 'TrackerMedianFlow_create'):
        return cv2.TrackerMedianFlow_create()
    if name == 'mosse' and hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    # Fallback: try legacy API
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        # Newer OpenCV versions (4.5+) sometimes require cv2.TrackerCSRT_create
        return None

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="camera index or video path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="device for YOLO")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (name or path)")
    ap.add_argument("--class-id", type=int, default=None, help="class id to track (None => any class). Example: 0 for person")
    ap.add_argument("--tracker", type=str, default="csrt", help="opencv tracker between detections (csrt,kcf,mosse)")
    ap.add_argument("--detect-every", type=int, default=10, help="run YOLO every N frames")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    args = ap.parse_args()

    # initialize detector & params
    yolo = YOLODetector(model_name=args.model, device=args.device, conf_threshold=args.conf)
    DETECT_EVERY = max(1, int(args.detect_every))
    tracker = None
    bbox = None
    frames_since_detect = DETECT_EVERY

    # video/camera
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = get_stream(src)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: cannot read source:", args.source)
        return

    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    # Kalman & PID
    kf = create_kalman()
    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)

    frame_idx = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detected_bbox = None

        # Decide whether to run detector
        if tracker is None or frames_since_detect >= DETECT_EVERY:
            # Run YOLO detection on this frame
            dets = yolo.detect(frame, classes=None if args.class_id is None else [args.class_id])
            # pick highest confidence detection (you can customize selection)
            if dets:
                best = max(dets, key=lambda d: d['conf'])
                detected_bbox = best['bbox']
                # (re)initialize tracker
                t = create_tracker_by_name(args.tracker)
                if t is not None:
                    try:
                        t.init(frame, tuple(detected_bbox))
                        tracker = t
                        bbox = detected_bbox
                    except Exception:
                        tracker = None
                        bbox = detected_bbox
                else:
                    tracker = None
                    bbox = detected_bbox
                frames_since_detect = 0
            else:
                # nothing detected
                frames_since_detect += 1
        else:
            # update tracker
            if tracker is not None:
                ok, tracked_box = tracker.update(frame)
                if ok:
                    x,y,w_box,h_box = [int(v) for v in tracked_box]
                    bbox = (x,y,w_box,h_box)
                    detected_bbox = bbox
                    frames_since_detect += 1
                else:
                    # tracker lost
                    tracker = None
                    bbox = None
                    frames_since_detect = DETECT_EVERY

        # If detection/tracking produced a bbox, correct kalman
        if detected_bbox is not None:
            x,y,w_box,h_box = detected_bbox
            cx = x + w_box/2.0
            cy = y + h_box/2.0
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            kf.correct(measurement)

        # Predict (smoothed position)
        pred = kf.predict()
        px = int(pred[0][0])
        py = int(pred[1][0])

        # Draw visuals
        if bbox:
            draw_bbox(frame, bbox)
            # use last measured center if available
            if detected_bbox is not None:
                draw_center(frame, int(cx), int(cy), color=(0,255,0))
        draw_center(frame, px, py, color=(255,0,0))       # Kalman
        draw_center(frame, cx_img, cy_img, color=(0,0,255)) # image center

        # PID control commands
        err_x = px - cx_img
        err_y = py - cy_img
        ctrl_x = pid_x.compute(err_x)
        ctrl_y = pid_y.compute(err_y)

        # overlay text: object name/conf if available
        info_text = f"ctrl_x={ctrl_x:.2f} ctrl_y={ctrl_y:.2f}"
        cv2.putText(frame, info_text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # FPS calc
        if frame_idx % 15 == 0:
            now = time.time()
            fps = 15.0 / (now - fps_time)
            fps_time = now
            cv2.putText(frame, f"FPS:{fps:.1f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("YOLO-Kalman-PID Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            # force detection on next frame
            frames_since_detect = DETECT_EVERY

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
