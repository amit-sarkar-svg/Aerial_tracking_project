# web_calib/backend/server.py
"""
FastAPI backend for Calibration Web App
- WebSocket /ws : receive base64 JPEG frames, run YOLO detection, return JSON
- POST /upload-frame : upload a single capture (data-url JSON) and save it to cal/images/cam_{id}/
- POST /calibrate-intrinsics : run calibrateCamera on saved captures for a camera
- POST /compute-extrinsics : compute extrinsic (rvec,tvec,cam_world_pos) from a chosen saved image
- GET /pdf : serves the uploaded PDF asset (path pointed to local upload)
"""

import os
import io
import json
import time
import base64
import shutil
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# YOLO (ultralytics)
from ultralytics import YOLO

# ----------------------
# Config / paths
# ----------------------
ROOT = Path(__file__).resolve().parent
CAL_IMG_DIR = ROOT / "cal" / "images"
CAL_DIR = ROOT / "cal"
OUTPUTS_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"
CAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
CAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Developer-provided PDF path (uploaded by you). Using the file you uploaded earlier:
PDF_SRC = "/mnt/data/Aerial Multi-Object & Multi-Camera Tracking System.pdf"
PDF_DST = STATIC_DIR / "Aerial_Multi-Object_&_Multi-Camera_Tracking_System.pdf"
if Path(PDF_SRC).exists() and not PDF_DST.exists():
    try:
        shutil.copy(PDF_SRC, PDF_DST)
    except Exception:
        pass

# Load YOLO model (will auto-download if missing)
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
print("[INFO] Loading YOLO model:", MODEL_NAME)
model = YOLO(MODEL_NAME)

# Homography store (camera id -> 3x3 matrix)
# On successful extrinsic/homography computation these files will be saved under web_calib/backend/cal/
HOMOGRAPHIES = {}

# Allow CORS from file:// or local frontend; adapt for production origins
app = FastAPI(title="Calibration Web Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Utility helpers
# ----------------------
def decode_data_url(data_url: str):
    """Accept either a data URL or raw base64 string. Return BGR image (numpy)."""
    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    b = base64.b64decode(encoded)
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def save_dataurl_to_file(data_url: str, out_path: Path):
    img = decode_data_url(data_url)
    if img is None:
        raise ValueError("Failed to decode image")
    cv2.imwrite(str(out_path), img)
    return out_path


def find_checkerboard_corners(gray, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    corners2 = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners2


def build_object_points(pattern_size, square_size_m):
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def save_json(path: Path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        return x
    def rec(o):
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rec(v) for v in o]
        return conv(o)
    with open(path, "w") as f:
        json.dump(rec(obj), f, indent=2)


# ----------------------
# WebSocket: realtime detection
# ----------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            text = await ws.receive_text()
            try:
                img = decode_data_url(text)
                if img is None:
                    await ws.send_text(json.dumps({"error": "invalid image"}))
                    continue
            except Exception as e:
                await ws.send_text(json.dumps({"error": "decode_failed", "detail": str(e)}))
                continue

            t0 = time.time()
            results = model.predict(img, imgsz=640, conf=0.35, verbose=False)
            t_inf = time.time() - t0

            # build simple list of detections
            dets = []
            r = results[0]
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                for b in r.boxes:
                    # handle torch tensors or numpy
                    xyxy = b.xyxy[0].cpu().numpy().tolist() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0].numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy().tolist()) if hasattr(b.conf[0], "cpu") else float(b.conf[0])
                    cls = int(b.cls[0].cpu().numpy().tolist()) if hasattr(b.cls[0], "cpu") else int(b.cls[0])
                    x1,y1,x2,y2 = [int(v) for v in xyxy]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dets.append({"bbox":[x1,y1,x2,y2], "class_id": cls, "conf": conf, "cx": cx, "cy": cy})
            payload = {"ts": time.time(), "inference_s": t_inf, "detections": dets}
            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("[WS] client disconnected")
    except Exception as e:
        print("[WS] error:", e)


# ----------------------
# HTTP endpoints
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/pdf")
async def get_pdf():
    if PDF_DST.exists():
        return FileResponse(str(PDF_DST), media_type="application/pdf", filename=PDF_DST.name)
    raise HTTPException(status_code=404, detail="pdf not found")


@app.post("/upload-frame")
async def upload_frame(camera_id: str = Form(...), data_url: str = Form(...)):
    """
    Upload a single captured frame (data URL) and save to cal/images/cam_{camera_id}/.
    Return saved path.
    """
    outdir = CAL_IMG_DIR / f"cam_{camera_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    out_path = outdir / f"{camera_id}_{timestamp}.jpg"
    try:
        save_dataurl_to_file(data_url, out_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"save_failed: {e}")
    return {"saved": str(out_path)}


@app.post("/calibrate-intrinsics")
async def calibrate_intrinsics_api(camera_id: str = Form(...), pattern_rows: int = Form(...),
                                   pattern_cols: int = Form(...), square_size_mm: float = Form(...),
                                   min_images: int = Form(8)):
    """
    Run intrinsic calibration using files in cal/images/cam_{camera_id}/.
    Saves outputs to outputs/camera_{id}_intrinsics.json
    """
    cam_dir = CAL_IMG_DIR / f"cam_{camera_id}"
    if not cam_dir.exists():
        raise HTTPException(status_code=404, detail="no captures for this camera")

    images = sorted(list(cam_dir.glob("*.jpg")))
    if len(images) < min_images:
        raise HTTPException(status_code=400, detail=f"not enough images ({len(images)}/{min_images})")

    pattern_size = (pattern_cols, pattern_rows)
    square_size_m = square_size_mm / 1000.0

    objp = build_object_points(pattern_size, square_size_m)
    objpoints = []
    imgpoints = []
    img_shape = None

    for p in images:
        img = cv2.imread(str(p))
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
            print("[CAL] corners not found in", p.name)

    if len(objpoints) < 6:
        raise HTTPException(status_code=400, detail=f"not enough valid corner detections ({len(objpoints)})")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    total_err = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += err
    mean_err = total_err / len(objpoints)

    intr_out = OUTPUTS_DIR / f"camera_{camera_id}_intrinsics.json"
    save_json(intr_out, {
        "camera_matrix": K,
        "distortion": dist,
        "reprojection_error_px": float(mean_err),
        "image_size": img_shape,
        "pattern_size": pattern_size,
        "square_size_m": square_size_m
    })

    return {"intrinsics": str(intr_out), "reproj_err_px": float(mean_err), "valid_images": len(objpoints)}


@app.post("/compute-extrinsics")
async def compute_extrinsics_api(camera_id: str = Form(...), image_name: str = Form(...),
                                 pattern_rows: int = Form(...), pattern_cols: int = Form(...),
                                 square_size_mm: float = Form(...)):
    """Compute extrinsic pose from a single image saved under cal/images/cam_{camera_id}/image_name"""
    cam_dir = CAL_IMG_DIR / f"cam_{camera_id}"
    image_path = cam_dir / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    intr_file = OUTPUTS_DIR / f"camera_{camera_id}_intrinsics.json"
    if not intr_file.exists():
        raise HTTPException(status_code=404, detail="intrinsics missing, run calibrate-intrinsics first")

    with open(intr_file, "r") as f:
        intr = json.load(f)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["distortion"], dtype=np.float64)

    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = (pattern_cols, pattern_rows)
    corners = find_checkerboard_corners(gray, pattern)
    if corners is None:
        raise HTTPException(status_code=400, detail="checkerboard corners not found in provided image")

    square_size_m = square_size_mm / 1000.0
    objp = build_object_points(pattern, square_size_m)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise HTTPException(status_code=500, detail="solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    cam_pos_world = (-R.T @ tvec).reshape(3)

    # save pose JSON
    pose_out = OUTPUTS_DIR / f"camera_{camera_id}_pose.json"
    save_json(pose_out, {
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "camera_position_world_m": cam_pos_world.tolist(),
        "image_used": str(image_path)
    })

    # Compute 4-corner homography (image -> world) using checkerboard corners
    # pick extreme indices (tl, tr, br, bl)
    pts = corners.reshape(-1, 2)
    cols, rows = pattern
    idx_tl = 0
    idx_tr = cols - 1
    idx_bl = (rows - 1) * cols
    idx_br = rows * cols - 1
    img_pts = np.array([pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]], dtype=np.float32)

    square = square_size_m
    world_pts = np.array([[0, 0],
                          [(cols - 1) * square, 0],
                          [(cols - 1) * square, (rows - 1) * square],
                          [0, (rows - 1) * square]], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)
    hom_out = CAL_DIR / f"cam{camera_id}_H.npy"
    np.save(str(hom_out), H)
    HOMOGRAPHIES[camera_id] = H

    return {"pose": str(pose_out), "homography": str(hom_out), "camera_position_world_m": cam_pos_world.tolist()}


# ----------------------
# Run with uvicorn
# ----------------------
if __name__ == "__main__":
    print("[INFO] Starting server on http://0.0.0.0:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
