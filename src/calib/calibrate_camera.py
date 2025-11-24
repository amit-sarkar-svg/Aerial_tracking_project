# src/calib/calibrate_camera.py
import cv2, json, glob, os, argparse, numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="glob or folder of images (e.g. cal/images/cam_0/*.png)")
    p.add_argument("--pattern-rows", type=int, default=6)
    p.add_argument("--pattern-cols", type=int, default=9)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--out", default="outputs/camera_intrinsics.json")
    return p.parse_args()

def main():
    args = parse_args()
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size = args.square_size_mm / 1000.0  # convert to meters
    # prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    images = sorted(glob.glob(args.images))
    if not images:
        raise SystemExit("No images found")

    objpoints = []
    imgpoints = []
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners_refined)
        else:
            print(f"[WARN] corners not found in {img_path}")

    if len(objpoints) < 8:
        raise SystemExit("Not enough valid calibration images (need >=8)")

    h, w = gray.shape[:2]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None,
                                                     flags=cv2.CALIB_RATIONAL_MODEL)
    # reprojection error
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
    mean_error = tot_error / len(objpoints)
    print(f"[INFO] Reprojection error: {mean_error:.6f} px")

    out = {
        "camera_matrix": K.tolist(),
        "distortion": dist.tolist(),
        "reprojection_error_px": float(mean_error),
        "image_size": [int(w), int(h)],
        "pattern_size": pattern_size,
        "square_size_m": float(square_size)
    }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVED] Intrinsics -> {args.out}")

if __name__ == "__main__":
    main()
