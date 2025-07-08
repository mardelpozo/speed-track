from abc import ABC, abstractmethod
import numpy as np

class Tracker(ABC):
    @abstractmethod
    def __init__(self, max_age: int, min_hits: int):
        pass

    @abstractmethod
    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        pass