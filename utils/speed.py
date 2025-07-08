import numpy as np
import json
from scipy.spatial.distance import euclidean

def estimate_speed(track: dict, homography: np.ndarray, fps: int):
    if len(track) < 2:
        return 0.0
    
    p1 = np.array([*track[0]["centroid"], 1.0])
    p2 = np.array([*track[-1]["centroid"], 1.0])

    w1 = homography @ p1
    w2 = homography @ p2
    w1 /= w1[2]
    w2 /= w2[2]

    dist = euclidean(w1[:2], w2[:2])
    dt = (track[-1]["frame_idx"] - track[0]["frame_idx"]) / fps

    return (dist / dt) * 3.6 if dt > 0 else 0.0

def load_homography(path: str):
    with open(path, "r") as f:
        return np.array(json.load(f))