import numpy as np
from trackers.base import Tracker
from yolox.tracker.byte_tracker import BYTETracker

class ByteTrack(Tracker):
    def __init__(self, config: dict, fps: int):
        self.tracker = BYTETracker(
            track_thresh=config["track_thresh"],
            track_buffer=config["track_buffer"],
            match_thresh=config["match_thresh"],
            frame_rate=fps,
        )

    def update(self, detections: list[dict]) -> list[dict]:
        dets = np.array([det["bbox"] + det["conf"], det["class_id"]] for det in detections)
        tracked = self.tracker.update(dets, [det["class_id"] for det in detections])
        results = []
        for track in tracked:
            xyxy = track[:4].astype(int).tolist()
            track_id = int(track[4])
            results.append({
                "id": track_id,
                "bbox": xyxy,
                "est_speed": None,
            })
        return results