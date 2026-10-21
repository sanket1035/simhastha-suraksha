"""Sample frames from a video file (stand-in for a live camera feed)."""
import cv2


def sample_frames(path, every_sec=2.0, max_frames=None):
    """Yield (time_in_seconds, frame_rgb) roughly every `every_sec` seconds."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_sec)))
    idx, yielded = 0, 0
    try:
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yielded += 1
                    if max_frames and yielded >= max_frames:
                        break
            idx += 1
    finally:
        cap.release()
