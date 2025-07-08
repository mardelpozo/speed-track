import numpy as np
from norfair import Detection, Tracker as NORFAIRTracker
from trackers.base import Tracker

def center(bbox: list[int]):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

class Norfair(Tracker):
    def __init__(self, config: dict):
        self.tracker = NORFAIRTracker(
            distance_function="euclidean",
            distance_threshold=30,
            initialization_delay=config["min_hits"],
            hit_counter_max=config["max_age"]
        )
        self.size = config["trackers"]["norfair"]["box_size"]

    def update(self, detections: list[dict]) -> list[dict]:
        dets = [Detection(points=np.array([center(det["bbox"])]), scores=np.array([det["conf"]])) for det in detections]
        tracked = self.tracker.update(dets)
        results = []
        for track in tracked:
            x, y = track.estimate[0]
            xyxy = [int(x-self.size), int(y-self.size), int(x+self.size), int(y+self.size)]
            results.append({
                "id": track.id,
                "bbox": xyxy,
                "est_speed": None,
            })
        return results