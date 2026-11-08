"""Training entrypoint for the GolfSwing event-sequencing model.

Training runs in two stages:
  1. the CNN backbone is frozen and only the LSTM head trains;
  2. the last FINE_TUNE_LAYERS CNN layers are unfrozen and everything trains
     at a lower learning rate.

The best model is chosen by PCE on whole validation videos (val_pce), not by
the frame-level event metrics.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/train.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from keras import callbacks, optimizers
from keras.metrics import SparseCategoricalAccuracy

from golfswing import metrics, model
from golfswing.callbacks import ValidationPCE
from golfswing.data import split_golfdb
from golfswing.dataset import (
    FullVideoGolfDataset,
    GolfDataset,
    make_full_video_tf_ds,
    make_tf_ds,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_ROOT = BASE_DIR / "logs"
MODEL_ROOT = BASE_DIR / "models"

# Data
SEED = 42            # fixed train/test split, so every run is scored on the same videos
CLIP_LEN = 64
BATCH = 5
LABEL_RADIUS = 1     # during training, frames this close to an event count as that event

# Model
UNITS = 256
BIDIRECTIONAL = True

# Stage 1: CNN frozen
HEAD_EPOCHS = 10
HEAD_LR = 1e-3

# Stage 2: fine-tune the top of the CNN
FINE_TUNE_EPOCHS = 40
FINE_TUNE_LAYERS = 30
FINE_TUNE_LR = 1e-4

# Validation / schedule
PCE_EVERY = 1             # full-video PCE is slow; raise to run it less often
LR_PATIENCE = 3           # epochs without val_pce improvement before halving the LR
EARLY_STOP_PATIENCE = 10  # epochs without improvement before a stage stops


def compile_model(cnn_lstm, learning_rate: float) -> None:
    """(Re)compile the model. Needed again after changing which layers train."""
    cnn_lstm.compile(
        loss=metrics.weighted_sparse_cce,
        metrics=[
            SparseCategoricalAccuracy(name="frame_accuracy"),
            metrics.EventPrecision(),
            metrics.EventRecall(),
            metrics.EventAUC(),
        ],
        optimizer=optimizers.Adam(learning_rate=learning_rate),
    )


def build_callbacks(run_id: str, log_root: Path, model_root: Path, val_videos):
    """Create the callbacks. The order matters: ValidationPCE must come first."""
    log_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    pce_cb = ValidationPCE(val_videos, clip_len=CLIP_LEN, every=PCE_EVERY)
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(model_root / f"{run_id}_best.weights.h5"),
        monitor="val_pce",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_pce",
        mode="max",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=1e-6,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_pce",
        mode="max",
        patience=EARLY_STOP_PATIENCE,
    )
    csv_logger = callbacks.CSVLogger(str(log_root / f"{run_id}.csv"), append=True)

    cbs = [pce_cb, checkpoint_cb, reduce_lr, early_stop, csv_logger]

    try:
        import tensorboard  # noqa: F401  (the callback needs it installed)
    except ImportError:
        print("TensorBoard is not installed; skipping TensorBoard logs.")
    else:
        cbs.append(callbacks.TensorBoard(
            log_dir=str(log_root / run_id),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
        ))

    return cbs


def main(max_videos=None, log_root: Path = LOG_ROOT, model_root: Path = MODEL_ROOT) -> None:
    dataframe = pd.read_pickle(DATA_DIR / "GolfDB.pkl")
    if max_videos:
        dataframe = dataframe.head(max_videos)
    train, test = split_golfdb(dataframe, seed=SEED)

    videos_dir = str(DATA_DIR / "videos_160")
    train_obj = GolfDataset(
        train, videos_dir, clip_len=CLIP_LEN, repeat_per_video=1,
        label_radius=LABEL_RADIUS,
    )
    test_obj = GolfDataset(
        test, videos_dir, clip_len=CLIP_LEN, train=False, repeat_per_video=1
    )

    train_ds = make_tf_ds(train_obj, batch_size=BATCH, clip_len=CLIP_LEN)
    test_ds = make_tf_ds(
        test_obj, batch_size=1, augment=False, shuffle=False, clip_len=CLIP_LEN
    )
    val_videos = make_full_video_tf_ds(FullVideoGolfDataset(test, videos_dir))

    steps_per_epoch = len(train_obj) // BATCH
    validation_steps = len(test_obj)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    cbs = build_callbacks(run_id, log_root, model_root, val_videos)
    checkpoint_cb = cbs[1]

    cnn_lstm = model.build_LSTM(
        clip_len=CLIP_LEN, units=UNITS, bidirectional=BIDIRECTIONAL
    )
    cnn_lstm.summary()

    def fit_stage(first_epoch: int, last_epoch: int) -> None:
        cnn_lstm.fit(
            train_ds,
            validation_data=test_ds,
            initial_epoch=first_epoch,
            epochs=last_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=cbs,
            shuffle=False,
            verbose="1",
        )

    # Stage 1: CNN frozen, train the LSTM head.
    model.set_cnn_trainable(cnn_lstm, 0)
    compile_model(cnn_lstm, HEAD_LR)
    fit_stage(0, HEAD_EPOCHS)

    # Stage 2: unfreeze the top of the CNN, lower learning rate.
    if FINE_TUNE_EPOCHS > 0:
        model.set_cnn_trainable(cnn_lstm, FINE_TUNE_LAYERS)
        compile_model(cnn_lstm, FINE_TUNE_LR)
        fit_stage(HEAD_EPOCHS, HEAD_EPOCHS + FINE_TUNE_EPOCHS)

    # Record how to rebuild this model (the web app and evaluation need these).
    settings = {
        "run_id": run_id,
        "weights": f"{run_id}_best.weights.h5",
        "best_val_pce": float(checkpoint_cb.best),
        "clip_len": CLIP_LEN,
        "units": UNITS,
        "bidirectional": BIDIRECTIONAL,
        "outputs": "logits",
        "label_radius": LABEL_RADIUS,
        "seed": SEED,
        "test_ids": [int(i) for i in test["id"]],
    }
    settings_path = model_root / f"{run_id}_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"Best val_pce {settings['best_val_pce']:.4f}. Settings saved to {settings_path}")


if __name__ == "__main__":
    main()
