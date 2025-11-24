# src/calib/capture_checkerboard.py
# Capture checkerboard images for a given camera source

import cv2
import os
import argparse
import time
from pathlib import Path

from src.calib.utils_calib import find_checkerboard_corners


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="Camera index or URL (e.g. 0 or http://192.168.1.101:8080/video)")
    p.add_argument("--outdir", default="cal/images",
                   help="Output directory (per camera subfolder auto-created)")
    p.add_argument("--camera-id", default="0",
                   help="Camera ID label for saved images")
    p.add_argument("--pattern-rows", type=int, default=6,
                   help="Inner checkerboard rows (6 for your 7-square board)")
    p.add_argument("--pattern-cols", type=int, default=7,
                   help="Inner checkerboard cols (7 for your 8-square board)")
    p.add_argument("--square-size-mm", type=float, default=25.0,
                   help="Checkerboard square size in millimeters")
    p.add_argument("--min-captures", type=int, default=25,
                   help="How many good captures to collect")
    return p.parse_args()


def open_capture(src):
    """
    Tries FFMPEG first (better for phone video).
    Falls back to default if unavailable.
    """
    try:
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    except Exception:
        cap = cv2.VideoCapture(src)
    return cap


def main():
    args = parse_args()

    outdir = Path(args.outdir) / f"cam_{args.camera_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"❌ ERROR: Cannot open source {args.source}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    collected = 0
    idx = 0

    print(f"[INFO] Capturing checkerboard images for camera {args.camera_id}")
    print(f"[INFO] Saving images to: {outdir}")
    print(f"[INFO] Need {args.min_captures} valid detections")

    while collected < args.min_captures:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, pattern_size)

        if corners is not None:
            # Draw detected pattern
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)

            fname = outdir / f"{args.camera_id}_{idx:03d}.png"
            cv2.imwrite(str(fname), vis)

            collected += 1
            idx += 1

            print(f"[OK] Saved {fname} ({collected}/{args.min_captures})")

            time.sleep(0.6)  # small delay improves unique captures

        # Show video
        try:
            cv2.imshow(f"Capture Cam {args.camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Capture completed for camera {args.camera_id}.")


if __name__ == "__main__":
    main()
