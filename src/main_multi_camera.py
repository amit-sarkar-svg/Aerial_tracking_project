# src/main_multi_camera.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import csv
from collections import deque, defaultdict
import numpy as np
import cv2

from src.utils.config import CAMERAS, FUSION_MODE, DEEPSORT_MAX_AGE, DEEPSORT_IOU_THRESHOLD, FUSED_LOG_PATH
from src.utils.multi_camera_stream import MultiCameraManager
from src.detectors.yolo_batch_detector import YOLOBatchDetector
from src.trackers.deep_sort.deep_sort import DeepSort
from src.utils.homography_utils import load_homography, transform_bbox_to_world
from src.utils.draw_utils import draw_bbox_id, draw_center
from src.trackers.kalman_filter import create_kalman_instance
from src.utils.object_sizes import OBJECT_WIDTHS, DEFAULT_OBJECT_WIDTH

def get_real_width_m(class_id):
    return OBJECT_WIDTHS.get(class_id, DEFAULT_OBJECT_WIDTH)

def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, 'a', newline='')
    w = csv.writer(f)
    if os.path.getsize(log_path) == 0:
        w.writerow(['timestamp','frame','track_id','class_id','class_name','world_x','world_y','distance_m'])
        f.flush()
    return f, w

def estimate_focal_from_frame_width(w_px):
    # heuristic focal px (useful for distance calculation if needed)
    fov_deg = 60.0
    return w_px / (2.0 * math.tan(math.radians(fov_deg)/2.0))

def main():
    # Initialize detector & streams
    yob = YOLOBatchDetector(model_name="yolov8n.pt", device="cpu", conf_threshold=0.35)
    cam_configs = CAMERAS
    # load homographies
    for c in cam_configs:
        hpath = c.get('homography', None)
        c['H'] = load_homography(hpath) if hpath is not None else None

    # Validate fusion mode vs homographies
    if FUSION_MODE == "homography":
        missing = [c['id'] for c in cam_configs if c.get('H') is None]
        if missing:
            print("[WARN] Fusion mode 'homography' selected but homography files missing for cameras:", missing)
            print("Falling back to 'independent' per-camera tracking. To enable fusion, compute homographies and set paths in src/utils/config.py")
            fusion_enabled = False
        else:
            fusion_enabled = True
    else:
        fusion_enabled = False

    cam_manager = MultiCameraManager(cam_configs)
    # Global DeepSort on world coords
    deepsort = DeepSort(max_age=DEEPSORT_MAX_AGE, iou_threshold=DEEPSORT_IOU_THRESHOLD)

    # per-track data
    track_kalmans = {}
    track_history = defaultdict(lambda: deque(maxlen=120))

    # logging
    fused_csv_file, fused_writer = init_csv(FUSED_LOG_PATH)

    frame_idx = 0
    print("[INFO] Multi-camera fusion started. Fusion enabled:", fusion_enabled)
    try:
        while True:
            frames = cam_manager.read_all()  # dict cid -> frame
            frame_idx += 1
            # collect per-camera detections and world-projected boxes
            fused_detections = []  # each item: dict with keys: world_bbox (x1,y1,x2,y2,cx,cy), conf, class_id, name, cam_id, img_bbox
            # Also keep per-camera display frames for drawing per-camera overlays (optional)
            display_frames = {}  # cam_id -> frame

            for cam in cam_configs:
                cid = cam['id']
                frame = frames.get(cid, None)
                display_frames[cid] = frame
                if frame is None:
                    continue
                dets = yob.detect(frame, classes=None)
                H = cam.get('H', None)
                for d in dets:
                    img_bbox = d['bbox']  # x,y,w,h
                    conf = d['conf']; cls = d['class_id']; name = d['name']
                    if fusion_enabled and H is not None:
                        world = transform_bbox_to_world(img_bbox, H)  # wx_min, wy_min, wx_max, wy_max, cx, cy
                        if world is None:
                            continue
                        wx_min, wy_min, wx_max, wy_max, wcx, wcy = world
                        fused_detections.append({
                            'world_bbox': (wx_min, wy_min, wx_max, wy_max),
                            'world_center': (wcx, wcy),
                            'conf': conf,
                            'class_id': cls,
                            'name': name,
                            'cam_id': cid,
                            'img_bbox': img_bbox
                        })
                    else:
                        # fusion disabled: we still append but mark no world coords
                        fused_detections.append({
                            'world_bbox': None,
                            'world_center': None,
                            'conf': conf,
                            'class_id': cls,
                            'name': name,
                            'cam_id': cid,
                            'img_bbox': img_bbox
                        })

            # If fusion enabled, create global detection list in world coordinates for DeepSort
            if fusion_enabled and len(fused_detections) > 0:
                # convert to DeepSort expected format: x1,y1,x2,y2,score,class_id,name
                ds_dets = []
                for fd in fused_detections:
                    if fd['world_bbox'] is None:
                        continue
                    wx1, wy1, wx2, wy2 = fd['world_bbox']
                    # For global tracking, we treat world bbox coords as a unified 2D plane
                    ds_dets.append((wx1, wy1, wx2, wy2, fd['conf'], fd['class_id'], fd['name']))
                tracks = deepsort.update_tracks(ds_dets)
                # tracks are in world coordinates
                # Now map tracks back to a representative image bbox for drawing and logging:
                for tr in tracks:
                    tid = tr.track_id
                    wx1, wy1, wx2, wy2 = tr.to_tlbr()
                    wcx = (wx1 + wx2) / 2.0
                    wcy = (wy1 + wy2) / 2.0
                    # find original detection(s) closest to track center to get camera id & img bbox
                    best = None; best_dist = float('inf')
                    for fd in fused_detections:
                        if fd['world_center'] is None:
                            continue
                        cx, cy = fd['world_center']
                        d = (cx - wcx)**2 + (cy - wcy)**2
                        if d < best_dist:
                            best = fd; best_dist = d
                    # now best contains representative detection with cam_id and img_bbox
                    rep_cam = best['cam_id'] if best is not None else None
                    rep_img_bbox = best['img_bbox'] if best is not None else None
                    class_id = tr.class_id if hasattr(tr, 'class_id') else (best['class_id'] if best is not None else -1)
                    class_name = tr.class_name if hasattr(tr, 'class_name') else (best['name'] if best is not None else "")
                    # ensure kalman
                    if tid not in track_kalmans:
                        track_kalmans[tid] = create_kalman_instance()
                    kf = track_kalmans[tid]
                    # use world center as measurement for kalman (2D world)
                    measurement = np.array([[wcx],[wcy]], dtype=np.float32)
                    try:
                        kf.correct(measurement)
                        pred = kf.predict()
                        px = float(pred[0][0]); py = float(pred[1][0])
                    except Exception:
                        px, py = wcx, wcy
                    # draw on representative camera frame if available
                    if rep_cam is not None and display_frames.get(rep_cam) is not None and rep_img_bbox is not None:
                        frame_draw = display_frames[rep_cam]
                        # draw bounding box and id (use draw_bbox_id)
                        x,y,w_box,h_box = rep_img_bbox
                        draw_bbox_id(frame_draw, (int(x),int(y),int(w_box),int(h_box)), tid, label=f"ID:{tid} {class_name}", distance=None, velocity=None, angle=None)
                        draw_center(frame_draw, px, py, color=(255,0,0))
                    # log track world center
                    ts = time.time()
                    fused_writer.writerow([ts, frame_idx, tid, class_id, class_name, float(wcx), float(wcy), ""])
                    fused_csv_file.flush()

            else:
                # Fusion disabled -> run per-camera independent DeepSort OR show independent tracks
                # For simplicity, we will run per-camera DeepSort separately (reuse DeepSort but instantiate per cam)
                # NOTE: This branch is fallback and not global fusion.
                # We'll draw per-camera detections and continue.
                for fd in fused_detections:
                    camid = fd['cam_id']
                    frame = display_frames.get(camid)
                    if frame is None:
                        continue
                    x,y,w_box,h_box = fd['img_bbox']
                    draw_bbox_id(frame, (int(x),int(y),int(w_box),int(h_box)), camid, label=f"Cam:{camid} {fd['name']}", distance=None, velocity=None, angle=None)

            # Display all camera frames (stack horizontally)
            # Build a simple tiled view of all cameras
            frames_to_show = []
            for cam in cam_configs:
                fid = cam['id']
                f = display_frames.get(fid)
                if f is None:
                    # placeholder black image
                    f = np.zeros((480, 640, 3), dtype=np.uint8)
                # resize to uniform small size
                f_small = cv2.resize(f, (640,480))
                cv2.putText(f_small, f"Camera {fid}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                frames_to_show.append(f_small)
            # tile in rows of 2
            row = None
            tiled = None
            per_row = 2
            for i in range(0, len(frames_to_show), per_row):
                group = frames_to_show[i:i+per_row]
                if len(group) == 1:
                    group.append(np.zeros_like(group[0]))
                row_img = np.hstack(group)
                if tiled is None:
                    tiled = row_img
                else:
                    tiled = np.vstack([tiled, row_img])
            if tiled is not None:
                cv2.imshow("Multi-Camera Fusion View", tiled)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        fused_csv_file.close()
        cam_manager.release_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
