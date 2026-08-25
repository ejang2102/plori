"""Video IO for organoid clips."""
import numpy as np
import cv2


def read_clip(path, scale=1.0, stride=1, max_frames=None):
    """Decode a clip to (T,H,W,3) uint8, downscaling each frame on the fly.
    Organoid videos are large (e.g. 2880x2048 x ~986 frames); use scale/stride/
    max_frames to track a couple of beat cycles at reduced resolution."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 reads BGR
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames)  # T,H,W,3  uint8


def video_fps(path, default=15.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    return fps
