# src/calib/capture_checkerboard.py
import cv2, os, argparse, time, json
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="camera index or URL (e.g. 0 or http://192.168.1.101:8080/video)")
    p.add_argument("--outdir", default="cal/images_cam", help="output directory (will create if missing)")
    p.add_argument("--camera-id", default="0", help="camera id label for filenames")
    p.add_argument("--pattern-rows", type=int, default=6, help="inner checkerboard rows (corners per row)")
    p.add_argument("--pattern-cols", type=int, default=9, help="inner checkerboard cols (corners per col)")
    p.add_argument("--square-size-mm", type=float, default=25.0, help="checkerboard square size in millimeters")
    p.add_argument("--min-captures", type=int, default=25, help="how many good captures to collect")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir) / f"cam_{args.camera_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source if not str(args.source).isdigit() else int(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    collected = 0
    idx = 0
    print(f"[INFO] Capturing checkerboard images to: {outdir} (need {args.min_captures})")
    while collected < args.min_captures:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received, retrying...")
            time.sleep(0.5)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # refine corners
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            fname = outdir / f"{args.camera_id}_{idx:03d}.png"
            # draw and save a preview image
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
            cv2.imwrite(str(fname), vis)
            collected += 1
            idx += 1
            print(f"[INFO] Saved {fname} ({collected}/{args.min_captures})")
            # small pause so you can reposition board
            time.sleep(0.8)
        # show live preview so you can move the board (optional in headful systems)
        cv2.imshow(f"Capture {args.camera_id}", frame)
        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Capture completed.")

if __name__ == "__main__":
    main()



'''
===>    # For laptop webcam index 0
python src/calib/capture_checkerboard.py --source 0 --camera-id 2 --outdir cal/images --min-captures 30

===>    # For phone IP stream
python src/calib/capture_checkerboard.py --source "http://192.168.1.101:8080/video" --camera-id 0 --outdir cal/images --min-captures 30
'''