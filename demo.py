import cv2
import supervision as sv
from utils.visualization import init_annotator, annotate_frame
from utils.speed import estimate_speed, load_homography

def run(video_path: str, detector: Detector, tracker: Tracker, homography: np.ndarray, speed_limit: float, device: str, output_path: str):
    annotator = init_annotator()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame)
        tracked = tracker.update(detections)

        for track in tracked:
            track_id = track["id"]
            bbox = track["bbox"]
            centroid = center(bbox)

            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append({
                "centroid": centroid,
                "frame_idx": frame_idx,
            })

            est_speed = estimate_speed(tracks[track_id], homography, fps)
            track["est_speed"] = est_speed

        annotated = annotate_frame(frame, tracked, annotator, speed_limit)

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
