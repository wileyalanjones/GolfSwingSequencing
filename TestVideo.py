import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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


def resize_video(vid_path, out_path, target_size=(160, 160)):
    """
    Resizes Shortest edge of new video to 160. 
    Long edge will be larger.

    """
    
    try:
        clip = VideoFileClip(vid_path)
        clip_resized = clip.resize(newsize=(target_size))
        clip_resized.write_videofile(out_path, codec="libc264")
        print(f"Video resized to {clip_resized.shape}")
        
        clip.close()
        clip_resized.close()
        
        return out_path
    
    except Exception as e:
        print(f"An error occured during processing: {e}")
        return None


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
        
        base, _ = os.path.splitext(self.path)
        
        self.resized_path = resize_video(
            self.path, 
            f"{base}_resized.mp4"
            )
        
        cap = cv2.VideoCapture(self.resized_path)
        frames = []
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.path}")
            return np.array([])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, _ = frame.shape
            target_h, target_w = self.target_size

            # only one axis will be cropped based on orientation of video
            start_h = max(0, (h - target_h) // 2)  
            end_h = start_h + target_h
            start_w = max(0, (w - target_w) // 2)
            end_w = start_w + target_w

            cropped_frame = frame[start_h:end_h, start_w:end_w]

            if cropped_frame.shape[0] != target_h or cropped_frame.shape[1] != target_w:
                cropped_frame = cv2.resize(cropped_frame, (target_w, target_h))

            frames.append(cropped_frame)

        cap.release()
        return np.array(frames)


