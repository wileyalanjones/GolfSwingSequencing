import cv2
import numpy as np
from moviepy import VideoFileClip
from keras.applications.mobilenet_v2 import preprocess_input


def compute_all_frame_predictions(model, frames, clip_len=64):
    """
    Gives max prediction for each Event in Video
    
    For Confustion Matrix
    """
    full_frames = frames.shape[0]
    preds_full = []

    for i in range(0, full_frames, clip_len):
        clip = frames[i:i+clip_len]
        original_clip_len = clip.shape[0]
        if original_clip_len < clip_len:
            pad_len = clip_len - original_clip_len
            pad_shape = (pad_len, clip.shape[1], clip.shape[2], clip.shape[3])
            clip = np.concatenate((clip, np.zeros(pad_shape, dtype=clip.dtype)), axis=0)

        clip = clip[None, ...]              # (1,clip_len,H,W,C)
        probs = model.predict(clip, verbose=0)[0]  # (clip_len,9)
        preds_full.append(probs[:original_clip_len])

    all_frame_probabilities = np.concatenate(preds_full, axis=0)
    
    max_indices = np.argmax(all_frame_probabilities, axis=1)
    
    return max_indices

def compute_event_frames(model, frames, clip_len=64, events=9):
    """
    Returns the frame numbers for the 8 Frames Events from Model

    For Unseen Video
    """
    full_frames = frames.shape[0]
    preds_full = []

    for i in range(0, full_frames, clip_len):
        clip = frames[i:i+clip_len]
        original_clip_len = clip.shape[0]
        if original_clip_len < clip_len:
            pad_len = clip_len - original_clip_len
            pad_shape = (pad_len, clip.shape[1], clip.shape[2], clip.shape[3])
            clip = np.concatenate((clip, np.zeros(pad_shape, dtype=clip.dtype)), axis=0)

        clip = clip[None, ...]              # (1,clip_len,H,W,C)
        probs = model.predict(clip, verbose=0)[0]  # (clip_len,9)
        preds_full.append(probs[:original_clip_len])

    all_frame_probabilities = np.concatenate(preds_full, axis=0)
    
    event_frames_indices = []
    for event_idx in range(1, events):
        frame_idx_for_event = np.argmax(all_frame_probabilities[:, event_idx])
        event_frames_indices.append(frame_idx_for_event)

    return np.array(event_frames_indices)


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
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
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
