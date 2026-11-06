"""Training entrypoint for the GolfSwing event-sequencing model."""

import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/train.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from golfswing import metrics, model
from golfswing.dataset import GolfDataset, make_tf_ds

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_ROOT = BASE_DIR / "logs"
MODEL_ROOT = BASE_DIR / "models"

CLIP_LEN = 16
BATCH = 5
NUM_CLASSES = 9
NUM_EPOCHS = 50


def build_callbacks(log_root: Path, model_root: Path) -> list:
    """Create the TensorBoard, checkpoint, LR-schedule, and CSV callbacks."""
    log_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    log_dir = log_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_template = (
        model_root
        / "model_128_CL16_B5_epoch_{epoch:02d}_val_event_auc_pr{val_event_auc_pr:.4f}.keras"
    )

    tensorboard_cb = callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
    )
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_template),
        monitor="val_event_auc_pr",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_event_auc_pr",
        factor=0.2,
        patience=1,
        min_lr=0.00001,
    )
    csv_logger = callbacks.CSVLogger(str(LOG_ROOT / "M256_B1_C16.csv"), append=True)

    return [tensorboard_cb, checkpoint_cb, reduce_lr, csv_logger]


def main() -> None:
    dataframe = pd.read_pickle(DATA_DIR / "GolfDB.pkl")
    train, test = train_test_split(dataframe)

    videos_dir = str(DATA_DIR / "videos_160")
    train_obj = GolfDataset(train, videos_dir, clip_len=CLIP_LEN, repeat_per_video=1)
    test_obj = GolfDataset(
        test, videos_dir, clip_len=CLIP_LEN, train=False, repeat_per_video=1
    )

    steps_per_epoch = len(train_obj) // BATCH
    validation_steps = len(test_obj)

    cnn_lstm = model.build_LSTM(clip_len=CLIP_LEN)
    cnn_lstm.summary()
    cnn_lstm.compile(
        loss=metrics.weighted_sparse_cce,
        metrics=[
            SparseCategoricalAccuracy(name="frame_accuracy"),
            metrics.EventPrecision(),
            metrics.EventRecall(),
            metrics.EventAUC(),
        ],
        optimizer="adam",
        run_eagerly=False,
    )

    cbs = build_callbacks(LOG_ROOT, MODEL_ROOT)

    for _ in range(NUM_EPOCHS):
        train_ds = make_tf_ds(train_obj, batch_size=BATCH, clip_len=CLIP_LEN)
        test_ds = make_tf_ds(
            test_obj, batch_size=1, augment=False, shuffle=False, clip_len=CLIP_LEN
        )
        cnn_lstm.fit(
            train_ds,
            validation_data=test_ds,
            epochs=1,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=cbs,
        )

    cnn_lstm.save(str(MODEL_ROOT / "Units128_CL16_B1.keras"))


if __name__ == "__main__":
    main()