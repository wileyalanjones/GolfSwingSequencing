"""Keras callbacks used during training."""

from keras import callbacks

from golfswing.metrics import compute_dataset_pce

EVENT_KEYS = [
    "address",
    "toe_up",
    "mid_backswing",
    "top",
    "mid_downswing",
    "impact",
    "mid_follow_through",
    "finish",
]


class ValidationPCE(callbacks.Callback):
    """Compute PCE on whole validation videos at the end of each epoch.

    Adds ``val_pce`` (overall) and ``val_pce_<event>`` (one per swing event)
    to the epoch logs. Put this callback BEFORE ModelCheckpoint,
    ReduceLROnPlateau, EarlyStopping and CSVLogger so they can see the values.

    videos : iterable of (frames, labels) for full videos, as built by
        ``make_full_video_tf_ds``.
    every  : run every N epochs (full-video evaluation is slow). Epochs that
        are skipped have no val_pce, so keep ``every=1`` if a monitor
        callback depends on it.
    """

    def __init__(self, videos, clip_len=64, stride=None, ordered=True, every=1):
        super().__init__()
        self.videos = videos
        self.clip_len = clip_len
        self.stride = stride
        self.ordered = ordered
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch % self.every != 0:
            return

        overall, per_event = compute_dataset_pce(
            self.model,
            self.videos,
            clip_len=self.clip_len,
            stride=self.stride,
            ordered=self.ordered,
        )

        logs["val_pce"] = float(overall)
        for key, value in zip(EVENT_KEYS, per_event):
            logs[f"val_pce_{key}"] = float(value)

        print(
            f"\nval_pce={overall:.4f}  "
            + "  ".join(f"{k}={v:.2f}" for k, v in zip(EVENT_KEYS, per_event))
        )
