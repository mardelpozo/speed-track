import numpy as np
import torch
from ultralytics import YOLO
from models.base import Detector

class YOLODetector(Detector):
    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25, iou: float = 0.5):
        self.model = YOLO(weights)
        self.device = device
        self.model.conf = conf
        self.model.iou = iou
        self.model.to(device)

    def predict(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(source=frame, device=self.device, verbose=False, stream=False)[0]
        
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            conf = float(box.conf.cpu().item())
            cls = int(box.cls.cpu().item())
            detections.append({
                "bbox": xyxy.tolist(),
                "conf": conf,
                "class_id": cls,
            })
            
        return detections