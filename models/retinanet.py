import cv2
import torch
import torchvision.transforms as T
from models.base import Detector
from torchvision.models.detection import retinanet_resnet50_fpn


class Retinanet(Detector):
    def __init__(self, device: str = "cpu", conf: float = 0.25, iou: float = 0.5):
        self.model = retinanet_resnet50_fpn(weights="COCO_V1", box_score_thresh=conf, box_nms_thresh=iou)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.conf = conf

    def predict(self, frame: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img)[0]

        detections = []
        for box, score, cls in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score < self.conf:
                continue

            detections.append({
                "bbox": [*map(int, box.cpu())],
                "conf": float(score.cpu()),
                "class_id": int(cls.cpu()),
            })
            
        return detections