import cv2
import torch
from models.base import Detector
from rfdetr import RFDetrBase

class RFDetr(Detector):
    def __init__(self, device: str = "cpu", conf: float = 0.25, iou: float = 0.5):
        self.model = RFDetrBase()
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.conf = conf

    def predict(self, frame: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb, threshold=self.conf)
        detections = []
        for (x1, y1, x2, y2), score, cls in zip(results["boxes"], results["scores"], results["class_ids"]):
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(score),
                "class_id": int(cls),
            })
            
        return detections