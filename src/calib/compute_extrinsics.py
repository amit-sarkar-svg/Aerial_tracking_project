# src/calib/compute_extrinsics.py
import cv2, json, argparse, numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--intrinsics", required=True, help="intrinsics json from calibrate_camera.py")
    p.add_argument("--image", required=True, help="image with checkerboard to compute pose")
    p.add_argument("--pattern-rows", type=int, default=6)
    p.add_argument("--pattern-cols", type=int, default=9)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--out", default="outputs/camera_pose.json")
    return p.parse_args()

def load_intrinsics(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    args = parse_args()
    intr = load_intrinsics(args.intrinsics)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["distortion"], dtype=np.float64).reshape(-1,1)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size = args.square_size_mm / 1000.0

    # object points in world coordinates (checkerboard located at world origin)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1,2)
    objp *= square_size

    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        raise SystemExit("Checkerboard corners not found in image.")

    corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Solve PnP: returns rvec, tvec such that objectPoints in world map to camera coordinate frame
    retval, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Camera pose: transform from world -> camera: X_cam = R * X_world + tvec
    # Inverse transform (camera position in world coords): X_world_cam = -R^T * tvec
    cam_pos_world = -R.T.dot(tvec).reshape(3)

    # Save results
    result = {
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "R": R.tolist(),
        "camera_position_world_m": cam_pos_world.tolist(),
        "intrinsics_file": args.intrinsics,
        "image_used": args.image
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Extrinsic pose -> {args.out}")
    print("Camera world position (meters):", cam_pos_world)

    # Optional: draw axis on image for verification
    axis_len = square_size * 3
    # project 3 axis points
    axis = np.float32([[axis_len,0,0],[0,axis_len,0],[0,0,-axis_len]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    corner = tuple(corners_refined[0].ravel().astype(int))
    img_vis = img.copy()
    cv2.line(img_vis, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)
    cv2.line(img_vis, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
    cv2.line(img_vis, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)
    cv2.imwrite(args.out.replace('.json','_axis.png'), img_vis)
    print("[INFO] axis visualization saved")

if __name__ == "__main__":
    main()
