# src/utils/multi_camera_stream.py
import cv2
import threading
import time

class CameraStream:
    """
    Simple threaded camera reader per source.
    """
    def __init__(self, source, name=None):
        self.source = source
        self.name = name or str(source)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.frame = frame if ret else None
            # small sleep to free CPU
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

class MultiCameraManager:
    def __init__(self, cam_configs):
        """
        cam_configs: list of dicts: {'id':int, 'source':..., 'homography':...}
        """
        self.streams = {}
        for c in cam_configs:
            cid = c['id']
            src = c['source']
            self.streams[cid] = CameraStream(src, name=f"cam{cid}")

    def read_all(self):
        """
        Returns dict: cid -> frame (or None)
        """
        frames = {}
        for cid, stream in self.streams.items():
            frames[cid] = stream.read()
        return frames

    def release_all(self):
        for s in self.streams.values():
            s.release()
