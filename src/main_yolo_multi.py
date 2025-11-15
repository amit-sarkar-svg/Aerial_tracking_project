# src/main_yolo_multi.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import csv
from collections import deque, defaultdict
import cv2
import numpy as np

# local imports
from src.detectors.yolo_detector import YOLODetector
from src.trackers.deep_sort.deep_sort import DeepSort
from src.trackers.kalman_filter import create_kalman_instance
from src.trackers.pid_controller import PID
from src.utils.draw_utils import draw_bbox_id, draw_center
from src.utils.object_sizes import OBJECT_WIDTHS, DEFAULT_OBJECT_WIDTH
from src.utils.graph_utils import draw_distance_graph
from src.utils.config import PID_KP, PID_KI, PID_KD
from src.utils.camera_stream import get_stream

# -----------------------
# helpers: distance & angle
# -----------------------
def get_real_width_m(class_id):
    return OBJECT_WIDTHS.get(class_id, DEFAULT_OBJECT_WIDTH)

def estimate_focal_length(focal_cfg, image_width):
    if focal_cfg and focal_cfg > 0:
        return float(focal_cfg)
    fov_deg = 60.0
    fov_rad = math.radians(fov_deg)
    return image_width / (2.0 * math.tan(fov_rad / 2.0))

def distance_from_pixels(real_width_m, focal_pixels, pixel_width):
    if pixel_width is None or pixel_width <= 0:
        return None
    return (real_width_m * focal_pixels) / float(pixel_width)

def angle_from_center(cx, cx_img, focal_pixels):
    return math.degrees(math.atan2((cx - cx_img), max(1.0, focal_pixels)))

# -----------------------
# CSV logging helpers
# -----------------------
def ensure_logs_dir(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

def init_csv(log_path):
    ensure_logs_dir(log_path)
    header = ['timestamp','frame','track_id','class_id','class_name','distance_m','velocity_m_s','angle_deg']
    new = not os.path.exists(log_path)
    f = open(log_path, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--focal", type=float, default=0.0, help="focal px optional")
    ap.add_argument("--log", type=str, default="src/logs/multi_log.csv")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--nms", type=float, default=0.45)
    args = ap.parse_args()

    # detector + tracker
    yolo = YOLODetector(model_name=args.model, device=args.device, conf_threshold=args.conf)
    deepsort = DeepSort(max_age=args.max_age)

    cap = get_stream(args.source)
    ret, frame = cap.read()
    if not ret:
        print("Cannot open source:", args.source)
        return

    h, w = frame.shape[:2]
    cx_img, cy_img = w//2, h//2
    focal = estimate_focal_length(args.focal if args.focal>0 else None, w)
    print(f"[INFO] focal px: {focal:.2f}")

    # per-track data stores
    track_kalmans = dict()            # track_id -> kalman instance
    track_last_distance = dict()      # track_id -> last distance
    track_last_time = dict()          # track_id -> last timestamp
    track_history = defaultdict(lambda: deque(maxlen=120))  # for graphs per track
    track_pids = dict()               # track_id -> (pid_x, pid_y)

    # logging
    csv_file, csv_writer = init_csv(args.log)

    frame_idx = 0
    fps_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO detect returns list of dicts: {'bbox':(x,y,w,h),'conf':..,'class_id':..,'name':..}
            detections = yolo.detect(frame, classes=None)
            # convert to deepsort format: [x1,y1,x2,y2,score,class_id,name]
            dets_for_ds = []
            for d in detections:
                x,y,w_box,h_box = d['bbox']
                x1=int(x); y1=int(y); x2=int(x+w_box); y2=int(y+h_box)
                dets_for_ds.append((x1,y1,x2,y2,d['conf'], d.get('class_id', -1), d.get('name','')))

            # DeepSort update: returns list of tracks with track_id, bbox, class_id, name
            tracks = deepsort.update_tracks(dets_for_ds, frame=frame)

            # collect per frame statuses to draw & log
            for tr in tracks:
                track_id = tr.track_id
                class_id = tr.class_id if tr.class_id is not None else -1
                class_name = tr.class_name if tr.class_name is not None else ""
                x1,y1,x2,y2 = tr.to_tlbr()
                w_box = x2 - x1
                h_box = y2 - y1
                cx = x1 + w_box/2.0
                cy = y1 + h_box/2.0

                # ensure kalman & pid exist
                if track_id not in track_kalmans:
                    track_kalmans[track_id] = create_kalman_instance()
                if track_id not in track_pids:
                    track_pids[track_id] = (PID(PID_KP, PID_KI, PID_KD), PID(PID_KP, PID_KI, PID_KD))

                kf = track_kalmans[track_id]
                pid_x, pid_y = track_pids[track_id]

                # measurement -> kalman correct
                measurement = np.array([[cx],[cy]], dtype=np.float32)
                kf.correct(measurement)
                pred = kf.predict()
                px = float(pred[0][0]); py = float(pred[1][0])

                # estimate distance, velocity, angle
                pixel_width = float(max(1.0, w_box))
                real_w = get_real_width_m(class_id)
                distance_m = distance_from_pixels(real_w, focal, pixel_width)
                now = time.time()
                velocity = None
                if track_id in track_last_distance and distance_m is not None:
                    dt = max(1e-6, now - track_last_time.get(track_id, now))
                    velocity = (distance_m - track_last_distance.get(track_id, distance_m)) / dt
                track_last_distance[track_id] = distance_m
                track_last_time[track_id] = now

                angle_deg = angle_from_center(cx, cx_img, focal) if distance_m is not None else None

                # update history for graph
                if distance_m is not None:
                    track_history[track_id].append(distance_m)

                # draw per-id bbox & labels
                label = f"ID:{track_id} {class_name}"
                draw_bbox_id(frame, (int(x1),int(y1),int(w_box),int(h_box)), track_id, label, distance=distance_m, velocity=velocity, angle=angle_deg)

                # draw predicted center and frame center
                draw_center(frame, int(px), int(py), color=(255,0,0))
                draw_center(frame, cx_img, cy_img, color=(0,0,255))

                # draw small graph for this track on right side (stacked)
                frame = draw_distance_graph(frame, list(track_history[track_id]), max_points=80)

                # CSV log: one row per track per frame
                csv_writer.writerow([now, frame_idx, track_id, class_id, class_name, f"{distance_m:.4f}" if distance_m else "", f"{velocity:.4f}" if velocity else "", f"{angle_deg:.4f}" if angle_deg else ""])
                csv_file.flush()

            # FPS overlay
            if frame_idx % 15 == 0:
                now = time.time()
                fps = 15.0 / (now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("YOLOv8 + DeepSORT Multi-Object Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        try:
            csv_file.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
