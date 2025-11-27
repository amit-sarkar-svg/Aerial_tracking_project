# src/calib/calibrate_camera.py
# Computes intrinsic camera parameters (K, distortion) from checkerboard images

import cv2
import json
import glob
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True,
                   help="Glob path for captured images (e.g. cal/images/cam_0/*.png)")
    p.add_argument("--pattern-rows", type=int, default=6,
                   help="Inner checkerboard rows (6 for your 7-square board)")
    p.add_argument("--pattern-cols", type=int, default=7,
                   help="Inner checkerboard cols (7 for your 8-square board)")
    p.add_argument("--square-size-mm", type=float, default=25.0,
                   help="Square size in millimeters")
    p.add_argument("--out", default="outputs/camera_intrinsics.json",
                   help="Output JSON file for intrinsics")
    return p.parse_args()


def main():
    args = parse_args()

    # Pattern size (cols, rows) — MUST match your printed board
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0  # convert mm → meters

    # Build object points (3D positions in the checkerboard coordinate system)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size_m

    images = sorted(glob.glob(args.images))
    if not images:
        raise SystemExit(f"❌ No images found for pattern: {args.images}")

    print(f"[INFO] Found {len(images)} images. Beginning intrinsic calibration...")

    objpoints = []
    imgpoints = []
    img_shape = None

    # Process each captured image
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Unable to read {img_path} — skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        # Find inner corners
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # Refine corner accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001
                )
            )

            objpoints.append(objp)
            imgpoints.append(corners_refined)
        else:
            print(f"[WARN] Corners not found in {img_path}")

    if len(objpoints) < 8:
        raise SystemExit("❌ Not enough valid calibration images (need ≥ 8).")

    # Perform calibration
    print("[INFO] Running OpenCV calibrateCamera()...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    # Compute reprojection error
    total_err = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += err

    mean_err = total_err / len(objpoints)
    print(f"[INFO] ✔ Calibration successful. Reprojection error = {mean_err:.4f} px")

    # Save JSON
    intrinsics = {
        "camera_matrix": K.tolist(),
        "distortion": dist.tolist(),
        "reprojection_error_px": float(mean_err),
        "image_size": list(img_shape),
        "pattern_size": list(pattern_size),
        "square_size_m": square_size_m
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(intrinsics, f, indent=2)

    print(f"[SAVED] Intrinsics → {args.out}")


if __name__ == "__main__":
    main()
