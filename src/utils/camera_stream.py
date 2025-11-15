# src/utils/camera_stream.py
import cv2

def get_stream(source=0):
    # source can be int (camera index) or path
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    # set some defaults to improve camera warmup
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
