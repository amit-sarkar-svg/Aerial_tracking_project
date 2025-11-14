# src/main_yolo.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import csv
from collections import deque
import cv2
import numpy as np

# local project imports
from src.utils.camera_stream import get_stream
from src.trackers.kalman_filter import create_kalman
from src.trackers.pid_controller import PID
from src.utils.draw_utils import draw_bbox, draw_center
from src.utils.config import PID_KP, PID_KI, PID_KD
from src.detectors.yolo_detector import YOLODetector
from src.utils.object_sizes import OBJECT_WIDTHS, DEFAULT_OBJECT_WIDTH
from src.utils.graph_utils import draw_distance_graph

# ----------------------------
# Helper: OpenCV tracker factory
# ----------------------------
def create_tracker_by_name(name='csrt'):
    name = (name or 'csrt').lower()
    if name == 'csrt' and hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if name == 'kcf' and hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if name == 'medianflow' and hasattr(cv2, 'TrackerMedianFlow_create'):
        return cv2.TrackerMedianFlow_create()
    if name == 'mosse' and hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    # fallback attempt for different OpenCV builds
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        return None

# ----------------------------
# Distance helpers
# ----------------------------
def get_real_width_m(class_id):
    return OBJECT_WIDTHS.get(class_id, DEFAULT_OBJECT_WIDTH)

def estimate_focal_length(focal_cfg, image_width):
    """
    Decide focal length in pixels.
    If user supplies focal_cfg (float > 0), use it.
    Otherwise, use a heuristic based on image width.
    """
    if focal_cfg is not None and focal_cfg > 0:
        return float(focal_cfg)
    # heuristic: assume horizontal field of view ~ 60 deg -> focal = width / (2*tan(fov/2))
    fov_deg = 60.0
    fov_rad = math.radians(fov_deg)
    focal = image_width / (2.0 * math.tan(fov_rad / 2.0))
    return focal

def distance_from_pixels(real_width_m, focal_pixels, pixel_width):
    """
    Distance (meters) from the pinhole camera model:
        Z = (RealWidth * Focal) / PixelWidth
    """
    if pixel_width is None or pixel_width <= 0:
        return None
    return (real_width_m * focal_pixels) / float(pixel_width)

def angle_from_center(cx, cx_img, focal_pixels):
    # horizontal angle in degrees: atan((cx - cx_img) / focal)
    return math.degrees(math.atan2((cx - cx_img), max(1.0, focal_pixels)))

# ----------------------------
# CSV logging helpers
# ----------------------------
def ensure_logs_dir(log_path):
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

def init_csv(log_path):
    ensure_logs_dir(log_path)
    header = ['timestamp', 'frame', 'class_id', 'class_name', 'distance_m', 'velocity_m_s', 'angle_deg']
    write_header = not os.path.exists(log_path)
    f = open(log_path, 'a', newline='')
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
        f.flush()
    return f, writer

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="camera index or video path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="device for YOLO")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (name or path)")
    ap.add_argument("--class-id", type=int, default=None, help="class id to restrict detection (None -> any)")
    ap.add_argument("--tracker", type=str, default="csrt", help="opencv tracker between detections (csrt,kcf,mosse)")
    ap.add_argument("--detect-every", type=int, default=10, help="run YOLO every N frames")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--focal", type=float, default=0.0, help="camera focal length in pixels (optional). If 0, heuristic used.")
    ap.add_argument("--log", type=str, default="src/logs/tracking_log.csv", help="CSV log path")
    args = ap.parse_args()

    # loader & config
    yolo = YOLODetector(model_name=args.model, device=args.device, conf_threshold=args.conf)
    DETECT_EVERY = max(1, int(args.detect_every))
    tracker = None
    bbox = None
    frames_since_detect = DETECT_EVERY

    # video/camera open
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

    # focal
    # try to load from focal_length.txt if present
    focal_pixels = None
    focal_file = os.path.join(os.path.dirname(__file__), "..", "focal_length.txt")
    if os.path.exists(focal_file):
        try:
            with open(focal_file, "r") as ff:
                val = float(ff.read().strip())
                if val > 0:
                    focal_pixels = val
        except Exception:
            focal_pixels = None
    focal_pixels = estimate_focal_length(args.focal if args.focal > 0 else focal_pixels, w)
    print(f"[INFO] Using focal (px): {focal_pixels:.2f}")

    # Kalman & PID
    kf = create_kalman()
    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)

    frame_idx = 0
    fps_time = time.time()

    # For velocity estimation (single tracked target)
    last_distance = None
    last_time = None

    # For graph overlay
    distance_history = deque(maxlen=160)

    # Auto-reloc variables
    lost_counter = 0
    AUTO_LOST_FRAMES = 8
    search_x, search_y = cx_img, cy_img
    forces_redetect = False

    # Logging
    log_path = args.log
    log_file, csv_writer = init_csv(log_path)

    tracked_class_id = None
    tracked_name = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detected_bbox = None
            detected_class_id = None
            detected_name = None

            # Decide whether to run detector
            if tracker is None or frames_since_detect >= DETECT_EVERY or forces_redetect:
                dets = yolo.detect(frame, classes=None if args.class_id is None else [args.class_id])
                if dets:
                    best = max(dets, key=lambda d: d['conf'])
                    detected_bbox = best['bbox']      # (x,y,w,h)
                    detected_class_id = best.get('class_id', None)
                    detected_name = best.get('name', None)
                    # reinit tracker
                    t = create_tracker_by_name(args.tracker)
                    if t is not None:
                        try:
                            t.init(frame, tuple(detected_bbox))
                            tracker = t
                        except Exception:
                            tracker = None
                    bbox = detected_bbox
                    frames_since_detect = 0
                    lost_counter = 0
                    forces_redetect = False
                    search_x, search_y = cx_img, cy_img
                    tracked_class_id = detected_class_id
                    tracked_name = detected_name
                else:
                    frames_since_detect += 1
            else:
                # tracker update
                if tracker is not None:
                    ok, tracked_box = tracker.update(frame)
                    if ok:
                        x, y, w_box, h_box = [int(v) for v in tracked_box]
                        bbox = (x, y, w_box, h_box)
                        detected_bbox = bbox
                        detected_class_id = tracked_class_id
                        detected_name = tracked_name
                        frames_since_detect += 1
                    else:
                        tracker = None
                        bbox = None
                        frames_since_detect = DETECT_EVERY

            # Kalman correct if measured
            if detected_bbox is not None:
                x, y, w_box, h_box = detected_bbox
                cx = x + w_box/2.0
                cy = y + h_box/2.0
                measurement = np.array([[cx], [cy]], dtype=np.float32)
                kf.correct(measurement)

            # Kalman predict
            pred = kf.predict()
            px = float(pred[0][0])
            py = float(pred[1][0])

            # lost handling
            if detected_bbox is None:
                lost_counter += 1
            else:
                lost_counter = 0
            is_lost = lost_counter > AUTO_LOST_FRAMES

            # Auto recenter & relock
            if is_lost:
                vx = float(kf.statePost[2])
                vy = float(kf.statePost[3])
                mag = max(1e-6, abs(vx) + abs(vy))
                vx, vy = vx / mag, vy / mag
                # move search point
                search_x += int(vx * 25)
                search_y += int(vy * 25)
                search_x = max(0, min(w - 1, search_x))
                search_y = max(0, min(h - 1, search_y))
                cv2.circle(frame, (search_x, search_y), 8, (0, 255, 255), -1)
                forces_redetect = True

            # distance, velocity, angle
            distance_m = None
            velocity_m_s = None
            angle_deg = None

            if detected_bbox is not None:
                _, _, w_box, h_box = detected_bbox
                pixel_width = float(w_box)
                # choose class width
                class_id = detected_class_id if detected_class_id is not None else tracked_class_id
                real_width_m = get_real_width_m(class_id) if class_id is not None else DEFAULT_OBJECT_WIDTH
                distance_m = distance_from_pixels(real_width_m, focal_pixels, pixel_width)
                # velocity using last measurement
                now = time.time()
                if last_distance is not None and last_time is not None and distance_m is not None:
                    dt = max(1e-6, now - last_time)
                    velocity_m_s = (distance_m - last_distance) / dt
                last_distance = distance_m
                last_time = now
                # angle
                cx = x + w_box/2.0
                angle_deg = angle_from_center(cx, cx_img, focal_pixels)
            # keep distance history for graph (use last known distance or None)
            if distance_m is not None:
                distance_history.append(distance_m)
            elif len(distance_history) > 0:
                # append last known to keep graph stable
                distance_history.append(distance_history[-1])
            else:
                # nothing to show
                pass

            # Draw visuals
            if bbox:
                draw_bbox(frame, bbox)
                if detected_bbox is not None:
                    draw_center(frame, int(cx), int(cy), color=(0,255,0))
            draw_center(frame, int(px), int(py), color=(255,0,0))   # kalman
            draw_center(frame, cx_img, cy_img, color=(0,0,255))     # frame center

            # PID
            err_x = px - cx_img
            err_y = py - cy_img
            ctrl_x = pid_x.compute(err_x)
            ctrl_y = pid_y.compute(err_y)

            # overlay status
            status_lines = []
            if tracked_name:
                status_lines.append(f"obj={tracked_name}:{tracked_class_id}")
            if distance_m is not None:
                status_lines.append(f"d={distance_m:.2f}m")
            if velocity_m_s is not None:
                status_lines.append(f"v={velocity_m_s:.2f}m/s")
            if angle_deg is not None:
                status_lines.append(f"ang={angle_deg:.1f}°")
            status_lines.append(f"ctrl_x={ctrl_x:.2f} ctrl_y={ctrl_y:.2f}")
            status_text = "  ".join(status_lines)
            cv2.putText(frame, status_text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # fps
            if frame_idx % 15 == 0:
                now = time.time()
                fps = 15.0 / (now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS:{fps:.1f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # lost flag
            cv2.putText(frame, f"Lost:{is_lost}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # draw distance graph
            frame = draw_distance_graph(frame, list(distance_history), max_points=120)

            # show
            cv2.imshow("YOLO-Kalman-PID Tracker (Distance+Velocity+Angle)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                frames_since_detect = DETECT_EVERY

            # CSV logging when we have a detection (log per frame if detected)
            if detected_bbox is not None:
                ts = time.time()
                row = [
                    ts,
                    frame_idx,
                    tracked_class_id,
                    tracked_name if tracked_name is not None else "",
                    f"{distance_m:.4f}" if distance_m is not None else "",
                    f"{velocity_m_s:.4f}" if velocity_m_s is not None else "",
                    f"{angle_deg:.4f}" if angle_deg is not None else "",
                ]
                try:
                    csv_writer.writerow(row)
                    log_file.flush()
                except Exception:
                    pass

    finally:
        try:
            log_file.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
