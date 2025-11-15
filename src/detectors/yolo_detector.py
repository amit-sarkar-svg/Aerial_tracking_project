# src/detectors/yolo_detector.py
# Minimal wrapper around ultralytics YOLO to return list of dicts:
# {'bbox': (x,y,w,h), 'conf': float, 'class_id': int, 'name': str}

from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", device="cpu", conf_threshold=0.35):
        self.model = YOLO(model_name)
        self.model.fuse()
        self.device = device
        self.conf = conf_threshold

    def detect(self, frame, classes=None):
        # ultralytics accepts numpy images
        results = self.model.predict(source=frame, conf=self.conf, device=self.device, verbose=False)
        # `results` is a list of Results (but when passing single frame returns list length 1)
        out = []
        if len(results) == 0:
            return out
        r = results[0]
        boxes = r.boxes  # Boxes object
        if boxes is None:
            return out
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, 'tolist') else float(box.conf)
            cls = int(box.cls[0]) if hasattr(box.cls, 'tolist') else int(box.cls)
            if (classes is not None) and (cls not in classes):
                continue
            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            x1,y1,x2,y2 = [int(v) for v in xyxy]
            w = x2 - x1
            h = y2 - y1
            name = self.model.names.get(cls, str(cls)) if hasattr(self.model, 'names') else str(cls)
            out.append({'bbox': (x1, y1, w, h), 'conf': conf, 'class_id': cls, 'name': name})
        return out
