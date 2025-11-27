# src/calib/auto_calibrate_all.py
# FULL AUTOMATED PIPELINE:
# capture -> intrinsics -> extrinsics -> homography -> summary -> (optional) fusion launch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import cv2
import numpy as np
import argparse
from pathlib import Path
import subprocess

from src.utils.config import CAMERAS
from src.calib.utils_calib import (
    find_checkerboard_corners,
    build_object_points,
    save_json
)


# --------------------------
# Helper: open capture
# --------------------------
def open_capture(source):
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap


# --------------------------
# Step 1: Capture checkerboard images (FIXED VERSION)
# --------------------------
def capture_checkerboard_images(cam_cfg, outdir, pattern_size, square_size_m, min_images=25):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    src = cam_cfg["source"]
    cap = open_capture(src)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open camera source: {src}")

    print(f"\n[INFO] Capturing for camera id={cam_cfg['id']} from {src}")
    print(f"[INFO] Saving to {outdir}")

    collected = 0

    while collected < min_images:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, pattern_size)

        if corners is not None:
            fname = Path(outdir) / f"{cam_cfg['id']}_{collected:03d}.png"

            # 🚀 FIX: Save exact frame where corners were detected
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imwrite(str(fname), vis)

            collected += 1
            print(f"[OK] Saved {fname} ({collected}/{min_images})")

            time.sleep(0.4)   # small delay to avoid duplicate frames

        # show viewer window
        try:
            cv2.imshow(f"Capturing cam {cam_cfg['id']}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()

    return list(sorted(Path(outdir).glob("*.png")))


# --------------------------
# Step 2: Intrinsics
# --------------------------
def calibrate_intrinsics(image_files, pattern_size, square_size_m):
    objp = build_object_points(pattern_size, square_size_m)

    objpoints = []
    imgpoints = []
    img_shape = None

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        corners = find_checkerboard_corners(gray, pattern_size)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"[WARN] Corners NOT found in {img_path}")

    if len(objpoints) < 8:
        raise RuntimeError("❌ Not enough valid calibration images (>= 8 required).")

    print("[INFO] Running calibrateCamera()...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )

    total_err = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += err

    mean_err = total_err / len(objpoints)

    return {
        "K": K,
        "dist": dist,
        "reproj_err": mean_err,
        "img_size": img_shape
    }


# --------------------------
# Step 3: Extrinsics
# --------------------------
def compute_extrinsic_from_image(intr, image_path, pattern_size, square_size_m):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"❌ Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = find_checkerboard_corners(gray, pattern_size)
    if corners is None:
        raise RuntimeError("❌ Checkerboard corners NOT found for extrinsics.")

    objp = build_object_points(pattern_size, square_size_m)
    K = intr["K"]
    dist = intr["dist"]

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        raise RuntimeError("❌ solvePnP failed.")

    R, _ = cv2.Rodrigues(rvec)
    cam_pos_world = -R.T @ tvec
    cam_pos_world = cam_pos_world.reshape(3)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "cam_pos_world": cam_pos_world,
        "image_used": str(image_path)
    }


# --------------------------
# Step 4: Homography
# --------------------------
def compute_homography_image_to_world(image_path, pattern_size, square_size_m):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = find_checkerboard_corners(gray, pattern_size)
    if corners is None:
        raise RuntimeError("❌ Cannot compute homography: corners not found.")

    pts = corners.reshape(-1, 2)
    cols, rows = pattern_size

    idx_tl = 0
    idx_tr = cols - 1
    idx_bl = (rows - 1) * cols
    idx_br = rows * cols - 1

    img_pts = np.array([
        pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]
    ], dtype=np.float32)

    square = square_size_m
    world_pts = np.array([
        [0, 0],
        [(cols - 1) * square, 0],
        [(cols - 1) * square, (rows - 1) * square],
        [0, (rows - 1) * square]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)
    return H


# --------------------------
# Step 5: Full Pipeline
# --------------------------
def run_for_all_cameras(args):
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0
    min_captures = args.min_captures

    summary = {}

    for cam in CAMERAS:
        cam_id = cam["id"]
        print(f"\n=== Processing CAMERA {cam_id} ===")

        images_outdir = Path("cal/images") / f"cam_{cam_id}"

        # 1. Capture
        image_files = capture_checkerboard_images(
            cam, str(images_outdir), pattern_size, square_size_m, min_images=min_captures
        )

        # 2. Intrinsics
        intr = calibrate_intrinsics(image_files, pattern_size, square_size_m)
        intr_out = Path("outputs") / f"camera_{cam_id}_intrinsics.json"
        save_json(intr_out, {
            "camera_matrix": intr["K"],
            "distortion": intr["dist"],
            "reprojection_error_px": float(intr["reproj_err"]),
            "image_size": intr["img_size"],
            "pattern_size": pattern_size,
            "square_size_m": square_size_m
        })
        print(f"[SAVED] Intrinsics → {intr_out}")

        # 3. Extrinsics
        pose_img = image_files[0]
        extr = compute_extrinsic_from_image(intr, pose_img, pattern_size, square_size_m)

        pose_out = Path("outputs") / f"camera_{cam_id}_pose.json"
        save_json(pose_out, {
            "rvec": extr["rvec"],
            "tvec": extr["tvec"],
            "R": extr["R"],
            "camera_position_world_m": extr["cam_pos_world"].tolist(),
            "image_used": extr["image_used"]
        })
        print(f"[SAVED] Pose → {pose_out}")

        # 4. Homography
        H = compute_homography_image_to_world(pose_img, pattern_size, square_size_m)
        H_out = Path("cal") / f"cam{cam_id}_H.npy"
        np.save(str(H_out), H)
        print(f"[SAVED] Homography → {H_out}")

        summary[cam_id] = {
            "intrinsics": str(intr_out),
            "pose": str(pose_out),
            "homography": str(H_out),
            "camera_position_world_m": extr["cam_pos_world"].tolist()
        }

    save_json("outputs/all_cameras_summary.json", summary)
    print("\n[INFO] Summary saved → outputs/all_cameras_summary.json")

    return summary


# --------------------------
# Argument Parser
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern-rows", type=int, default=6)
    p.add_argument("--pattern-cols", type=int, default=7)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--min-captures", type=int, default=25)
    p.add_argument("--launch-fusion", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    args = parse_args()

    Path("cal").mkdir(exist_ok=True)
    Path("cal/images").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    summary = run_for_all_cameras(args)

    if args.launch_fusion:
        print("[INFO] Launching fusion...")
        subprocess.run([
            "python",
            "src/main_multi_camera.py",
            "--fusion", "homography",
            "--device", args.device
        ])
