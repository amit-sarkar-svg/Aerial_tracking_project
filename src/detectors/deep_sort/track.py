# src/trackers/deep_sort/track.py
class Track:
    def __init__(self, track_id, tlbr, class_id=None, class_name=""):
        self.track_id = int(track_id)
        self._tlbr = tuple(map(float, tlbr))
        self.class_id = class_id
        self.class_name = class_name
        self.age = 0  # frames since last matched
    def to_tlbr(self):
        return self._tlbr
    def update(self, tlbr, class_id=None, class_name=""):
        self._tlbr = tuple(map(float, tlbr))
        if class_id is not None:
            self.class_id = class_id
        if class_name:
            self.class_name = class_name
