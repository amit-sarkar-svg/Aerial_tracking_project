# src/detectors/yolo_detector.py
"""
Simple YOLOv8 detector wrapper using ultralytics.
- model: path or model name (e.g., 'yolov8n.pt' or 'yolov8n')
- device: 'cpu' or 'cuda'
- detect(frame) -> returns list of bboxes (x,y,w,h), confidences, class_ids
"""

from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', device='cpu', conf_threshold=0.35, iou=0.45):
        # This will automatically download the model if not present
        self.model = YOLO(model_name)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou = iou

    def detect(self, frame, classes=None):
        """
        Detect objects in a single frame.
        Args:
            frame: BGR image (numpy)
            classes: list of class IDs to keep (None -> keep all)
        Returns:
            detections: list of dicts: {'bbox': (x,y,w,h), 'conf': float, 'class_id': int, 'name':str}
        """
        # Ultralyics expects RGB
        img = frame[:, :, ::-1]
        # run inference (fast mode)
        results = self.model.predict(img, conf=self.conf_threshold, iou=self.iou, verbose=False)
        detections = []

        # results is a list (one element per image); we used single image
        if len(results) == 0:
            return detections

        r = results[0]
        # r.boxes contains boxes; boxes.xyxy, boxes.conf, boxes.cls
        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            return detections

        xyxy = r.boxes.xyxy.cpu().numpy()        # (N,4) x1,y1,x2,y2
        confs = r.boxes.conf.cpu().numpy()       # (N,)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # filter by class if requested
        for (x1,y1,x2,y2), conf, cid in zip(xyxy, confs, cls_ids):
            if classes is not None and cid not in classes:
                continue
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            w = x2 - x1
            h = y2 - y1
            detections.append({
                'bbox': (int(x1), int(y1), int(w), int(h)),
                'conf': float(conf),
                'class_id': int(cid),
                'name': self.model.names.get(int(cid), str(cid)) if hasattr(self.model, 'names') else str(cid)
            })
        return detections
