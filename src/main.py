import sys, os
import cv2
import numpy as np

# Fix Python import path so "src" folder works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.camera_stream import get_stream
from src.detectors.color_detector import detect_color
from src.trackers.kalman_filter import create_kalman
from src.trackers.pid_controller import PID
from src.utils.config import PID_KP, PID_KI, PID_KD
from src.utils.draw_utils import draw_bbox, draw_center


def main():
    # Open webcam, or replace with "videos/sample_test.mp4"
    cap = get_stream(0)

    ret, frame = cap.read()
    if not ret:
        print("Camera could not start.")
        return

    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    # Initialize Kalman + PID
    kalman = create_kalman()
    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Color-based detection
        bbox, mask = detect_color(frame)

        if bbox is not None:
            x, y, w_box, h_box = bbox
            cx = x + w_box / 2
            cy = y + h_box / 2

            # FIX: use numpy instead of cv2.UMat
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            kalman.correct(measurement)

        # Kalman prediction (always works even without detection)
        prediction = kalman.predict()

        # FIX: extract values correctly
        px = int(prediction[0][0])
        py = int(prediction[1][0])

        # DRAWING
        if bbox:
            draw_bbox(frame, bbox)
            draw_center(frame, cx, cy, color=(0, 255, 0))  # detected center

        draw_center(frame, px, py, color=(255, 0, 0))       # Kalman prediction
        draw_center(frame, cx_img, cy_img, color=(0, 0, 255))  # image center

        # PID CONTROL
        error_x = px - cx_img
        error_y = py - cy_img

        ctrl_x = pid_x.compute(error_x)
        ctrl_y = pid_y.compute(error_y)

        # Text overlay
        cv2.putText(
            frame,
            f"ctrl_x={ctrl_x:.2f}  ctrl_y={ctrl_y:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Tracking System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
