#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:51:30 2025

@author: wileyjones
"""

import tensorflow as tf
import numpy as np

CLASS_WEIGHTS = tf.constant([1.] + [35/8.]*8, dtype=tf.float32)

scce = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE)

def weighted_sparse_cce(y_true, y_pred):
    """
        Parameters
    ----------
    y_true : Li
        DESCRIP.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_true = tf.cast(y_true, tf.int32)
    base_loss = scce(y_true, y_pred)
    weights = tf.gather(CLASS_WEIGHTS, y_true)
    weighted_loss = base_loss * weights
    
    return tf.reduce_mean(weighted_loss)

class EventPrecision(tf.keras.metrics.Precision):
    """
    Override of Precision to be event or non-event
    """
    def __init__(self, name="event_precision"):
        super().__init__(name=name, thresholds=.2)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, tf.int32)
        y_true_bin = tf.cast(tf.greater(y_true, 0), tf.int32)
        
        prob_back = y_pred[..., 0]
        prob_event = 1.0 - prob_back
        
        return super().update_state(y_true_bin, prob_event, sample_weight)

class EventRecall(tf.keras.metrics.Recall):
    """
    Override of Recall to be event or non-event
    """
    def __init__(self, name="event_recall"):
        super().__init__(name=name, thresholds=.2)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, tf.int32)
        y_true_bin = tf.cast(tf.greater(y_true, 0), tf.int32)
        
        prob_back = y_pred[..., 0]
        prob_event = 1.0 - prob_back
        
        return super().update_state(y_true_bin, prob_event, sample_weight)
    
class EventAUC(tf.keras.metrics.AUC):
    """
    AUC over event vs background.
    """
    def __init__(self, name="event_auc_pr", curve="PR", **kwargs):
        super().__init__(name=name, curve=curve, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true_bin = tf.cast(tf.greater(y_true, 0), tf.float32)

        prob_back = y_pred[..., 0]
        prob_event = 1.0 - prob_back   # shape (...,)

        return super().update_state(y_true_bin, prob_event, sample_weight)
    
def correct_preds(probs, labels, tol=-1):
    """
    Determine which event predictions are correct within a tolerance.

    Args:
        probs: np.array of shape (T, 9) softmax probabilities over classes
        labels: np.array of shape (T,) integer labels per frame
        tol: int, tolerance in frames. If -1, compute automatically.

    Returns:
        events: np.array of true event frame indices, shape (8,)
        preds: np.array of predicted event frame indices, shape (8,)
        deltas: np.array of abs errors per event, shape (8,)
        tol: int, tolerance used
        correct: np.array of 0/1 correctness per event, shape (8,)
    """

    # true event frame indices for classes 1..8
    events = np.where((labels > 0) & (labels < 9))[0]
    preds = np.zeros(len(events), dtype=np.int32)

    if len(events) == 0:
        # no events found in labels => return empty correctness
        return events, preds, np.array([]), tol, np.array([])

    # auto tolerance: based on frames between first and last event
    if tol == -1:
        # Original paper: tol = max(round((impact - address)/30), 1)
        # Here events[0] should be Address, events[5] should be Impact (index 5 in 8 events)
        tol = int(max(np.round((events[5] - events[0]) / 30), 1))

    # predicted event frame for each event class i (1..8)
    # class i corresponds to column i in probs
    for i in range(len(events)):
        preds[i] = np.argmax(probs[:, i + 1])  # +1 because class 0 is background

    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)

    return events, preds, deltas, tol, correct


def compute_pce_for_video(model, frames_np, labels_np, clip_len=32):
    """
    Run model over an entire video (frames_np) in clip_len chunks,
    then compute PCE for that video.

    Args:
        model: trained keras model
        frames_np: np.array (T, H, W, 3) float32 frames (already resized)
        labels_np: np.array (T,) int labels
        clip_len: int, how many frames per forward pass

    Returns:
        pce_video: float, mean correctness over 8 events
        correct: np.array shape (8,), per-event correctness
        extras: (events, preds, deltas, tol)
    """

    T = frames_np.shape[0]
    probs_full = []

    # run through full video in chunks
    for start in range(0, T, clip_len):
        end = min(start + clip_len, T)
        clip = frames_np[start:end][None, ...]  # shape (1, t, H, W, C)
        probs = model.predict(clip, verbose=0)[0]  # shape (t, 9)
        probs_full.append(probs)

    probs_full = np.concatenate(probs_full, axis=0)  # (T, 9)

    events, preds, deltas, tol, correct = correct_preds(probs_full, labels_np)

    if correct.size == 0:
        return 0.0, correct, (events, preds, deltas, tol)

    pce_video = float(np.mean(correct))
    return pce_video, correct, (events, preds, deltas, tol)


def compute_dataset_pce(model, full_video_dataset, clip_len=32):
    """
    Compute mean PCE over a dataset of full videos.

    full_video_dataset should yield:
        frames_np: (T, H, W, 3)
        labels_np: (T,)

    Returns:
        pce_mean: float, average PCE across videos
        per_event_acc: np.array (8,), average accuracy per event across videos
    """

    all_correct = []
    for frames_np, labels_np in full_video_dataset:
        pce_video, correct, _ = compute_pce_for_video(
            model, frames_np, labels_np, clip_len=clip_len
        )
        if correct.size > 0:
            all_correct.append(correct)

    if not all_correct:
        return 0.0, np.zeros(8)

    all_correct = np.stack(all_correct, axis=0)  # (N_videos, 8)

    pce_mean = float(np.mean(all_correct))
    per_event_acc = np.mean(all_correct, axis=0)  # (8,)

    return pce_mean, per_event_acc
