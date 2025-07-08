from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    @abstractmethod
    def __init__(self, weights: str, device: str, conf_thresh: float, iou_thresh: float):
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[dict]:
        pass