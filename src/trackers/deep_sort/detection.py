# src/trackers/deep_sort/detection.py
class Detection:
    def __init__(self, x1,y1,x2,y2,score=1.0, class_id=None, class_name=""):
        self.x1 = float(x1); self.y1=float(y1); self.x2=float(x2); self.y2=float(y2)
        self.score = float(score)
        self.class_id = class_id
        self.class_name = class_name

    def to_tlbr(self):
        return (self.x1, self.y1, self.x2, self.y2)
