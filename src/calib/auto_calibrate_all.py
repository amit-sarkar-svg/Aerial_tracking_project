# src/calib/auto_calibrate_all.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import cv2
import numpy as np
import argparse
from pathlib import Path
import subprocess
from src.utils.config import CAMERAS
from src.calib.utils_calib import find_checkerboard_corners, build_object_points, save_json

def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # prefer ffmpeg backend
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap

def capture_checkerboard_images(cam_cfg, outdir, pattern_size, square_size_m, min_images=25, timeout_per_attempt=10):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    src = cam_cfg['source']
    cap = open_capture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {src}")
    print(f"[INFO] Capturing for camera id={cam_cfg['id']} from {src} -> saving to {outdir}")
    collected = 0
    attempt = 0
    start_time = time.time()
    while collected < min_images:
        ret, frame = cap.read()
        if not ret:
            attempt += 1
            if attempt > 50:
                raise RuntimeError(f"[ERROR] No frames received from {src}. Check URL and network.")
            time.sleep(0.2)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, pattern_size)
        if corners is not None:
            fname = Path(outdir) / f"{cam_cfg['id']}_{collected:03d}.png"
            # save original frame for later calibrate (with board drawn)
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imwrite(str(fname), vis)
            collected += 1
            print(f"[INFO] Saved {fname} ({collected}/{min_images})")
            time.sleep(0.6)  # small pause for reposition
        # show preview if GUI available
        try:
            cv2.imshow(f"Capturing cam {cam_cfg['id']}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            # headless environment: ignore imshow
            pass
    cap.release()
    cv2.destroyAllWindows()
    return list(sorted(Path(outdir).glob("*.png")))

def calibrate_intrinsics(image_files, pattern_size, square_size_m):
    # object points
    objp = build_object_points(pattern_size, square_size_m)
    objpoints = []
    imgpoints = []
    img_shape = None
    for p in image_files:
        img = cv2.imread(str(p))
        if img_shape is None:
            img_shape = img.shape[:2][::-1]  # w,h
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, pattern_size)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"[WARN] Corners not found in {p}, skipping")
    if len(objpoints) < 8:
        raise RuntimeError("Not enough valid calibration images (need >= 8)")
    print("[INFO] Running cv2.calibrateCamera() ...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None,
                                                     flags=cv2.CALIB_RATIONAL_MODEL)
    # reprojection error
    tot_error = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        tot_error += err
    mean_err = tot_error / len(objpoints)
    return {"K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "reproj_err": mean_err, "img_size": img_shape}

def compute_extrinsic_from_image(intrinsics, image_path, pattern_size, square_size_m):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = find_checkerboard_corners(gray, pattern_size)
    if corners is None:
        raise RuntimeError("Checkerboard corners not found in pose image")
    objp = build_object_points(pattern_size, square_size_m)
    K = intrinsics["K"]
    dist = intrinsics["dist"]
    retval, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    cam_pos_world = -R.T.dot(tvec).reshape(3)
    return {"rvec": rvec, "tvec": tvec, "R": R, "cam_pos_world": cam_pos_world, "image_used": str(image_path)}

def compute_homography_image_to_world(image_path, pattern_size, square_size_m):
    # compute homography mapping image pixels -> world XY plane (Z=0) using corners of outer 4 corner points
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = find_checkerboard_corners(gray, pattern_size)
    if corners is None:
        raise RuntimeError("Checkerboard corners not found for homography")
    # corners are ordered left->right top->bottom for inner corners
    # pick the four outer corner image points (top-left, top-right, bottom-right, bottom-left) from corners array
    cols, rows = pattern_size
    pts = corners.reshape(-1,2)
    idx_tl = 0
    idx_tr = cols - 1
    idx_bl = (rows - 1) * cols
    idx_br = rows * cols - 1
    img_pts = np.array([pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]], dtype=np.float32)
    # corresponding world XY (meters), Z=0
    square = square_size_m
    world_pts = np.array([
        [0.0, 0.0],
        [(cols-1)*square, 0.0],
        [(cols-1)*square, (rows-1)*square],
        [0.0, (rows-1)*square]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, world_pts, 0)
    return H

def run_for_all_cameras(args):
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0
    min_captures = args.min_captures
    # iterate CAMERAS from config
    results = {}
    for cam in CAMERAS:
        cam_id = cam['id']
        print(f"\n=== Processing camera id={cam_id} source={cam['source']} ===")
        images_outdir = Path("cal/images") / f"cam_{cam_id}"
        image_files = capture_checkerboard_images(cam, str(images_outdir), pattern_size, square_size_m, min_images=min_captures)
        # calibrate intrinsics
        intr = calibrate_intrinsics(image_files, pattern_size, square_size_m)
        # save intrinsics
        intr_out = Path("outputs") / f"camera_{cam_id}_intrinsics.json"
        save_json(intr_out, {
            "camera_matrix": intr["K"],
            "distortion": intr["dist"],
            "reprojection_error_px": float(intr["reproj_err"]),
            "image_size": intr["img_size"],
            "pattern_size": pattern_size,
            "square_size_m": square_size_m
        })
        print(f"[SAVED] Intrinsics -> {intr_out}")
        # compute extrinsics using first saved image (pose image)
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
        print(f"[SAVED] Pose -> {pose_out}")
        # compute homography image->world plane
        H = compute_homography_image_to_world(pose_img, pattern_size, square_size_m)
        H_out = Path("cal") / f"cam{cam_id}_H.npy"
        np.save(str(H_out), H)
        print(f"[SAVED] Homography -> {H_out}")
        # store summary
        results[cam_id] = {
            "intrinsics": str(intr_out),
            "pose": str(pose_out),
            "homography": str(H_out),
            "camera_position": extr["cam_pos_world"].tolist()
        }
    # write a combined summary
    save_json("outputs/all_cameras_summary.json", results)
    print("[INFO] All cameras processed. Summary saved -> outputs/all_cameras_summary.json")
    return results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern-rows", type=int, default=6, help="checkerboard inner rows")
    p.add_argument("--pattern-cols", type=int, default=9, help="checkerboard inner cols")
    p.add_argument("--square-size-mm", type=float, default=25.0, help="square size in mm")
    p.add_argument("--min-captures", type=int, default=25, help="good checkerboard images per camera")
    p.add_argument("--launch-fusion", action="store_true", help="launch main_multi_camera.py after calibration")
    p.add_argument("--device", type=str, default="cpu", help="device argument passed to main_multi_camera")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # ensure output directories
    Path("cal").mkdir(exist_ok=True)
    Path("cal/images").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    # run
    summary = run_for_all_cameras(args)
    print("[INFO] Calibration complete for all cameras.")
    if args.launch_fusion:
        # update config CAMERAS to point to new homographies (optional)
        # We will just call the main script and let it load cal/camX_H.npy paths from config.py
        cmd = ["python", "src/main_multi_camera.py", "--fusion", "homography", "--device", args.device]
        print("[INFO] Launching multi-camera fusion:", " ".join(cmd))
        subprocess.run(cmd, check=False)
