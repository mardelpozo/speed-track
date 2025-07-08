import supervision as sv
import numpy as np
import cv2

def init_annotator():
    return sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color.white()
    )

def annotate_frame(frame: np.ndarray, detections: list[dict], annotator: sv.BoxAnnotator, speed_limit: float):
    dets = sv.Detections.from_xyxy([det["bbox"] for det in detections])

    labels = []
    colors = []

    for det in detections:
        speed = det["est_speed"]
        label = f"ID {det['id']}  {speed:.1f} km/h"
        labels.append(label)

        if speed > speed_limit:
            colors.append(sv.Color.red())
        else:
            colors.append(sv.Color.white())

    annotated = annotator.annotate(
        scene=frame.copy(),
        detections=dets,
        labels=labels,
        color=colors
    )
    return annotated