import random
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

def softmax(logits, axis=-1):
    """
    Turn raw model scores (logits) into probabilities along ``axis``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)  # numerical stability
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def ordered_event_frames(logits):
    """
    Pick one frame per event so the frames are in strictly increasing order
    and the total score is as high as possible.

    Parameters
    ----------
    logits : array of shape (T, 1 + K)
        Per-frame scores. Column 0 is the "no event" class and is ignored;
        columns 1..K are the K events, in swing order.

    Returns
    -------
    np.array of K ints
        Frame index for each event.
    """
    scores = np.asarray(logits, dtype=np.float64)[:, 1:]
    T, K = scores.shape

    if T < K:  # not enough frames to order them, fall back to independent picks
        return np.argmax(scores, axis=0)

    positions = np.arange(T)
    best = np.full((K, T), -np.inf)       # best[k, t]: best total with event k at frame t
    back = np.zeros((K, T), dtype=np.int64)
    best[0] = scores[:, 0]

    for k in range(1, K):
        prev = best[k - 1]
        run_max = np.maximum.accumulate(prev)                                     # best of event k-1 up to t
        run_arg = np.maximum.accumulate(np.where(prev == run_max, positions, 0))  # where it happened
        best[k, 1:] = scores[1:, k] + run_max[:-1]                                # event k-1 must be earlier
        back[k, 1:] = run_arg[:-1]

    frames = np.zeros(K, dtype=np.int64)
    frames[-1] = int(np.argmax(best[-1]))
    for k in range(K - 1, 0, -1):
        frames[k - 1] = back[k, frames[k]]

    return frames

def get_events_dataframe(dataframe, num_events=8):
    """
    Parameters
    ----------
    dataframe : Pandas DataFrame 
        Train or Test dataset to extract event frames to ID.
    num_events : Integer
        Number of event framse. The default is 8.

    Returns
    -------
    id_to_events : Dictionary
        Keys = video 1, values = event frame list.

    """
    id_to_events = {}
    
    for _, row in dataframe.iterrows():
        vid = row['id']
        events = row['events'][1:-1]
        assert len(events) == num_events, f"video {vid}: only {len(events)}"
        id_to_events[vid] = events
        
    return id_to_events
   
     
def get_num_frames(vid_path: str):
    """
    Parameters
    ----------
    vid_path : str
        Path to video from directory.

    Raises
    ------
    RuntimeError
        Error if video path cannot be opened with CV2.

    Returns
    -------
    count : int
        Total number of frames in video clip.

    """
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {vid_path}")
        
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return count 

def get_random_clip_indexes(
        num_frames, 
        clip_len,
        events=None,
        buffer=20):
    """
    Parameters
    ----------
    num_frames : Integer
        Total Number of frames in video.
    clip_len : Integer
        Total number of frames for final clip segemetn.

    Returns
    -------
    List
        Frames numbers to capture for training.

    """
    if events is None:
        start = random.randint(0, max(0, num_frames - clip_len))
        return start + np.arange(clip_len)

    events = np.asarray(events, dtype=np.int32)  

    start_buffer = max(int(events[0]) - buffer, 0)
    end_buffer   = min(int(events[-1]) + buffer, num_frames - clip_len)

    if end_buffer < start_buffer:
        start = random.randint(0, max(0, num_frames - clip_len))
    else:
        start = random.randint(start_buffer, end_buffer)

    return start + np.arange(clip_len)

def get_frames_by_idxs(vid_path, idxs):
    """
    Parameters
    ----------
    vid_path : str
        Path to video for frame extraction.
    idxs : List
        List of frames to capture for training purposes.

    Raises
    ------
    RuntimeError
        If CV2 is not able to capture frame from video. 

    Returns
    -------
    np.array
        Final Tensor with (T, H, W, C)
        T = number of frames in clip
        H and W = Height and Width of frame
        C = color channels.

    """
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {vid_path}")
    
    frames = []
    
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise RuntimeError(f"Read failed at frame {idx} in {vid_path}")

        # Must convert to RGB to work with ImageNet
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()

    return np.stack(frames, axis=0)


def get_clip_labels(idxs, events, radius=0):
    """
    Parameters
    ----------
    idxs : List
        clips numbers to match labels to frames.
    events :  List
        Event frame list.
    radius : Integer
        Frames within ``radius`` of an event also get that event's label
        (the nearest event wins). 0 labels only the exact event frame.

    Returns
    -------
    np.array
        Labels of frames per clip. (T, ) 
        T = Number of Frames per clip 

    """
    events = np.asarray(events, dtype=np.int64)
    labels = np.zeros(len(idxs), dtype=np.int32)

    for i, idx in enumerate(idxs):
        dist = np.abs(events - int(idx))
        nearest = int(np.argmin(dist))
        if dist[nearest] <= radius:
            labels[i] = nearest + 1

    return labels

def get_clip_weights(labels, background=1, event=1):
    """
    Parameters
    ----------
    labels : TYPE
        DESCRIPTION.
    background : TYPE, optional
        DESCRIPTION. The default is 1.
    event : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    weights = np.where(labels==0, background, event)
    return np.array(weights, dtype=np.float32) 

        
        
