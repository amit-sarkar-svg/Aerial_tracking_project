import cv2
import math

print("\n=== AUTO CAMERA CALIBRATION TOOL ===")
print("1️⃣ Place a known-width object in front of camera.")
print("2️⃣ Enter its real width (meters) and distance (meters).")
print("3️⃣ Draw a box around the object.\n")

real_width = float(input("Enter real object width (meters): "))
distance = float(input("Enter distance from camera (meters): "))

cap = cv2.VideoCapture(0)

print("\nPress 's' to capture frame…")

while True:
    ret, frame = cap.read()
    cv2.imshow("Calibration - Press S", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()

print("\nDraw the bounding box around object:")
r = cv2.selectROI("Select ROI", frame, False)
cv2.destroyAllWindows()

pixel_width = r[2]

focal = (pixel_width * distance) / real_width
print(f"\n📌 Estimated focal length (pixels): {focal:.2f}")

with open("focal_length.txt", "w") as f:
    f.write(str(focal))

print("Saved to focal_length.txt")
