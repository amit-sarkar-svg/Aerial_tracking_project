# src/trackers/deep_sort/deep_sort.py
import numpy as np
from collections import deque
from .track import Track
from .detection import Detection
from .nn_matching import iou_cost_matrix, linear_assignment

class DeepSort:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = dict()   # track_id -> Track
        self._next_id = 1

    def update_tracks(self, detections, frame=None):
        """
        detections: list of tuples (x1,y1,x2,y2,score,class_id,name)
        returns list of Track objects (active)
        """
        dets = [Detection(x1,y1,x2,y2,score,class_id,name) for (x1,y1,x2,y2,score,class_id,name) in detections]

        # if no existing tracks, create from all detections
        if len(self.tracks) == 0:
            for d in dets:
                tr = Track(self._next_id, d.to_tlbr(), d.class_id, d.class_name)
                self.tracks[self._next_id] = tr
                self._next_id += 1
            return list(self.tracks.values())

        # build matrices for cost (IoU)
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t].to_tlbr() for t in track_ids]
        det_boxes = [d.to_tlbr() for d in dets]

        if len(track_boxes)>0 and len(det_boxes)>0:
            cost = iou_cost_matrix(track_boxes, det_boxes)
            matches, unmatched_tr, unmatched_det = linear_assignment(cost, thresh=1-self.iou_threshold)
        else:
            matches = []
            unmatched_tr = list(range(len(track_ids)))
            unmatched_det = list(range(len(det_boxes)))

        # update matched
        for t_idx, d_idx in matches:
            tid = track_ids[t_idx]
            d = dets[d_idx]
            self.tracks[tid].update(d.to_tlbr(), d.class_id, d.class_name)
            self.tracks[tid].age = 0

        # create new tracks for unmatched detections
        for d_idx in unmatched_det:
            d = dets[d_idx]
            tr = Track(self._next_id, d.to_tlbr(), d.class_id, d.class_name)
            self.tracks[self._next_id] = tr
            self._next_id += 1

        # age unmatched tracks and delete old
        for t_idx in unmatched_tr:
            tid = track_ids[t_idx]
            self.tracks[tid].age += 1
        # remove tracks older than max_age
        to_del = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return list(self.tracks.values())
