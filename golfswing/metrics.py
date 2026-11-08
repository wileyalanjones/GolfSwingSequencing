import tensorflow as tf
import numpy as np
from typing import Literal

from golfswing.predict import predict_logits
from golfswing.utils import ordered_event_frames


CLASS_WEIGHTS = tf.constant([1.] + [35/8.]*8, dtype=tf.float32)

scce = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)

def weighted_sparse_cce(y_true, y_pred):
    """
    Weight Sparse Crossentropy Loss Function.
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
        probs = tf.nn.softmax(y_pred, axis=-1)  # model outputs logits
        prob_back = tf.gather(probs, 0, axis=-1)
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

        probs = tf.nn.softmax(y_pred, axis=-1)  # model outputs logits
        prob_back = tf.gather(probs, 0, axis=-1)
        prob_event = 1.0 - prob_back
        
        return super().update_state(y_true_bin, prob_event, sample_weight)
    
class EventAUC(tf.keras.metrics.AUC):
    """
    AUC over event vs background.
    """
    def __init__(self, name="event_auc_pr", curve: Literal["ROC", "PR"] = "PR", **kwargs):
        super().__init__(name=name, curve=curve, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true_bin = tf.cast(tf.greater(y_true, 0), tf.float32)

        probs = tf.nn.softmax(y_pred, axis=-1)  # model outputs logits
        prob_back = tf.gather(probs, 0, axis=-1)
        prob_event = 1.0 - prob_back

        return super().update_state(y_true_bin, prob_event, sample_weight)
    
def correct_preds(scores, labels, tol=-1, ordered=True):
    """
    Determines if Model predictions are correct within tolerance

    ``scores`` are the per-frame model scores, shape (T, 9). With
    ordered=True the 8 event frames are picked together in swing order (the
    same way the app picks them); otherwise each event is picked on its own.
    """

    events = np.where((labels > 0) & (labels < 9))[0]
    preds = np.zeros(len(events), dtype=np.int32)

    if len(events) == 0:
        # no events found in labels => return empty correctness
        return events, preds, np.array([]), tol, np.array([])

    if tol == -1:
        tol = int(max(np.round((events[5] - events[0]) / 30), 1))

    if ordered and len(events) == scores.shape[1] - 1:
        preds = ordered_event_frames(scores).astype(np.int32)
    else:
        for i in range(len(events)):
            preds[i] = np.argmax(scores[:, i + 1])

    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)

    return events, preds, deltas, tol, correct


def compute_pce_for_video(model, frames_np, labels_np, clip_len=64,
                          stride=None, ordered=True):
    """
    Gets Model scores for the whole video (overlapping clip_len windows)
    and feeds them to correct preds
    """

    # TF -> numpy
    if hasattr(frames_np, "numpy"): frames_np = frames_np.numpy()
    if hasattr(labels_np, "numpy"): labels_np = labels_np.numpy()

    scores = predict_logits(model, frames_np, clip_len=clip_len, stride=stride)
    events, preds, deltas, tol, correct = correct_preds(scores, labels_np, ordered=ordered)

    if correct.size == 0:
        return 0.0, correct, (events, preds, deltas, tol)

    return float(np.mean(correct)), correct, (events, preds, deltas, tol)


def compute_dataset_pce(model, full_video_dataset, clip_len=64,
                        stride=None, ordered=True):
    """
    Gives the total PCE for all events and per event
    """

    all_correct = []
    for frames_np, labels_np in full_video_dataset:
        pce_video, correct, _ = compute_pce_for_video(
            model, frames_np, labels_np, clip_len=clip_len,
            stride=stride, ordered=ordered
        )
        if correct.size > 0:
            all_correct.append(correct)

    if not all_correct:
        return 0.0, np.zeros(8)

    all_correct = np.stack(all_correct, axis=0)
    
    return float(np.mean(all_correct)), np.mean(all_correct, axis=0)
