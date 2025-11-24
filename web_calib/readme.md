# Calibration Web App (branch: feature/calibration-web)

Purpose
-------
This folder contains the scaffold for a browser-based calibration UI (capture frames for checkerboard calibration). Backend integration (FastAPI) will be added in a later commit/PR.

How to use (quick)
------------------
1. Checkout branch:
   git checkout feature/calibration-web

2. Open the frontend:
   Open file `web_calib/frontend/index.html` in Chrome (or any modern browser).

3. Camera permission:
   Allow camera access, point the camera to the checkerboard, press "Capture" to save a JPEG image locally.

Notes
-----
- Keep this branch isolated; no changes are made to `src/` or to `main`.
- The official design doc for the project is attached as an asset:
  `Aerial Multi-Object & Multi-Camera Tracking System.pdf` (local path):
  /mnt/data/Aerial Multi-Object & Multi-Camera Tracking System.pdf

Next steps
----------
After you test the frontend capture page, I will:
1. Add backend WebSocket + YOLO server (FastAPI)
2. Add server-side calibration endpoints
3. Add homography + extrinsic generation pipeline
4. Add CSV logging and download

