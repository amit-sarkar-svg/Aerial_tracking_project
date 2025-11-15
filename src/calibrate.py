# src/calibrate.py
import cv2
import math
import os

print("\n=== AUTO CAMERA CALIBRATION TOOL ===")
print("Place a known-width object at a known distance from the camera.")
real_width = float(input("Enter real object width (meters): "))
distance = float(input("Enter distance from camera (meters): "))

cap = cv2.VideoCapture(0)
print("\nPress 's' to capture a frame...")

frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available")
        break
    cv2.imshow("Calibration - Press S", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

if frame is None:
    raise SystemExit("No frame captured.")

print("\nDraw the bounding box around object and press ENTER.")
r = cv2.selectROI("Select ROI", frame, False)
cv2.destroyAllWindows()
pixel_width = r[2]
focal = (pixel_width * distance) / real_width
print(f"\nEstimated focal length (pixels): {focal:.2f}")

out_file = os.path.join(os.path.dirname(__file__), "..", "focal_length.txt")
with open(out_file, "w") as f:
    f.write(str(focal))
print(f"Saved to {out_file}")
