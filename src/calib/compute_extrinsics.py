# src/calib/compute_extrinsics.py
# Computes camera extrinsics (R, t, world position) from a single checkerboard image.

import cv2
import json
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--intrinsics", required=True,
                   help="Intrinsics JSON from calibrate_camera.py")
    p.add_argument("--image", required=True,
                   help="Image showing checkerboard (must contain full board)")
    p.add_argument("--pattern-rows", type=int, default=6,
                   help="Inner rows (6 for your 7-square board)")
    p.add_argument("--pattern-cols", type=int, default=7,
                   help="Inner columns (7 for your 8-square board)")
    p.add_argument("--square-size-mm", type=float, default=25.0,
                   help="Square size in millimeters")
    p.add_argument("--out", default="outputs/camera_pose.json",
                   help="Output JSON for extrinsics")
    return p.parse_args()


def load_intrinsics(path):
    """Load intrinsics from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    # Load intrinsics
    intr = load_intrinsics(args.intrinsics)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["distortion"], dtype=np.float64)

    # Pattern spec
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0

    # Build known 3D object points (Z=0 plane)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size_m

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"❌ ERROR: Cannot read image {args.image}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        raise SystemExit("❌ ERROR: Checkerboard corners not found in the image.")

    # Refine corner positions
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    # SolvePnP → extrinsics
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        raise SystemExit("❌ solvePnP failed.")

    # Convert rvec → rotation matrix R
    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world coordinates:
    # cam_pos = -R^T * t
    cam_pos_world = -R.T @ tvec
    cam_pos_world = cam_pos_world.reshape(3)

    # Save JSON
    result = {
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "R": R.tolist(),
        "camera_position_world_m": cam_pos_world.tolist(),
        "intrinsics_file": args.intrinsics,
        "image_used": args.image,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[SAVED] Extrinsic pose → {args.out}")
    print(f"[INFO] Camera world position (meters): {cam_pos_world}")

    # ---- Visualization of XYZ axes ----
    axis_len = square_size_m * 3
    axis = np.float32([
        [axis_len, 0, 0],    # X = red
        [0, axis_len, 0],    # Y = green
        [0, 0, -axis_len]    # Z = blue (negative because image coords)
    ])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    corner = tuple(corners_refined[0].ravel().astype(int))

    img_vis = img.copy()
    cv2.line(img_vis, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 4)
    cv2.line(img_vis, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 4)
    cv2.line(img_vis, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 4)

    out_img = args.out.replace(".json", "_axes.png")
    cv2.imwrite(out_img, img_vis)

    print(f"[INFO] Axis visualization saved → {out_img}")


if __name__ == "__main__":
    main()
