import cv2
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input

from golfswing.utils import ordered_event_frames


def predict_logits(model, frames, clip_len=64, stride=None, batch_size=4):
    """
    Runs the model over a whole video and returns one row of raw scores
    (logits) per frame, shape (T, num_classes).

    The video is cut into windows of ``clip_len`` frames that overlap by
    ``clip_len - stride`` frames. A frame covered by several windows gets the
    average of their scores, so frames near a window edge (where the LSTM has
    less context) are not decided by that edge alone.

    stride=None uses half a window. stride=clip_len gives plain
    non-overlapping chunks.
    """
    frames = np.asarray(frames)
    total = frames.shape[0]
    if stride is None:
        stride = max(clip_len // 2, 1)

    if total <= clip_len:
        starts = [0]
    else:
        starts = list(range(0, total - clip_len + 1, stride))
        if starts[-1] != total - clip_len:   # make sure the last frames are covered
            starts.append(total - clip_len)

    score_sum = None
    counts = np.zeros(total)

    for b in range(0, len(starts), batch_size):
        batch_starts = starts[b:b + batch_size]
        windows = []
        for s in batch_starts:
            window = frames[s:s + clip_len]
            if window.shape[0] < clip_len:   # video shorter than one window: zero-pad
                pad = np.zeros((clip_len - window.shape[0], *window.shape[1:]), dtype=window.dtype)
                window = np.concatenate([window, pad], axis=0)
            windows.append(window)

        logits = model.predict(np.stack(windows), verbose=0, batch_size=len(windows))

        if score_sum is None:
            score_sum = np.zeros((total, logits.shape[-1]))
        for s, out in zip(batch_starts, logits):
            n = min(clip_len, total - s)
            score_sum[s:s + n] += out[:n]
            counts[s:s + n] += 1

    assert score_sum is not None
    return score_sum / counts[:, None]


def compute_all_frame_predictions(model, frames, clip_len=64, stride=None):
    """
    Gives the highest-scoring class for every frame in the video.
    
    For Confustion Matrix
    """
    all_frame_logits = predict_logits(model, frames, clip_len, stride)

    return np.argmax(all_frame_logits, axis=1)


def compute_event_frames(model, frames, clip_len=64, events=9, ordered=True, stride=None):
    """
    Returns the frame numbers for the 8 Frames Events from Model

    For Unseen Video

    ordered=True picks the 8 frames together so they are in swing order with
    the highest total score. ordered=False picks each event's best frame
    independently.
    """
    all_frame_logits = predict_logits(model, frames, clip_len, stride)[:, :events]

    if ordered:
        return ordered_event_frames(all_frame_logits)

    return np.argmax(all_frame_logits[:, 1:], axis=0)

def extract_frames(video_path, frame_indices):
    """
    Grabs full-resolution frames from the original video.
    Returns a list of images (BGR numpy arrays), one per frame index
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")
    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        images.append(frame)
    cap.release()
    return images


class TestVideo:
    """
    Object for Unseen Test Video for Model
    """

    def __init__(self, path, target_size=(160, 160)):
        self.path = path
        self.target_size = target_size
        self.frames = preprocess_input(self._load_and_crop_frames())
        print(f"Loaded frames shape: {self.frames.shape}")
        

    def _load_and_crop_frames(self):
        """
        Prepares video for model. Resize, crop, and capture. 
        """
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {self.path}")

        target_h, target_w = self.target_size
        pad_color = (124, 116, 104)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            scale = min(target_h / h, target_w / w)
            new_w, new_h = round(w * scale), round(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            frame = cv2.copyMakeBorder(
                frame, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=pad_color
            )
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
