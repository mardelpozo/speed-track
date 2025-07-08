import os 
import glob
import cv2
import json
import numpy as np
import logging
from tqdm import tqdm

def parse_gt(path: str):
    with open(path, "r") as f:
        line = f.readline().strip()
        speed, timestamp = map(float, line.split())
    return speed, timestamp

def frame_timestamp(frame_idx: int, fps: int):
    return frame_idx / fps

def load_dataset(config: dict):
    video_dir = config["dataset"]["path"]
    video_glob = config["dataset"]["video_glob"]
    fps = config["video"]["fps"]
    skip_frames = config["video"]["skip_frames"]
    show_progress = config["logging"]["progress"]
    verbosity = config["logging"]["verbosity"]

    video_paths = sorted(glob.glob(os.path.join(video_dir, video_glob)))
    video_iter = tqdm(video_paths, desc="Loading dataset", unit="video", disable=not show_progress)

    for video_path in video_iter:
        basename = os.path.splitext(os.path.basename(video_path))[0]
        txt_path = os.path.join(video_dir, f"{basename}.txt")

        if not os.path.exists(txt_path):
            logging.warning(f"Annotation file {txt_path} not found, skipping {video_path}")
            continue

        gt_speed, gt_timestamp = parse_gt(txt_path)
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            timestamp = frame_timestamp(frame_idx, fps)
            yield {
                "video_name": basename,
                "frame": frame,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "gt_speed": gt_speed,
                "gt_timestamp": gt_timestamp,
            }
            frame_idx += 1

        cap.release()