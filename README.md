# Golf Swing Sequencing

A deep learning model that detects the 8 key events of a golf swing (address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish) directly from raw swing video, trained on the [GolfDB](https://github.com/wmcnally/golfdb) dataset.

## How it works

The model treats swing-event detection as per-frame classification over a video clip:

- **CNN backbone** — a frozen `MobileNetV2` (ImageNet weights) extracts features from each `160x160` frame.
- **Temporal head** — a `TimeDistributed` wrapper applies the CNN to every frame in a clip, then a (bidirectional) LSTM models motion across time.
- **Output** — a `TimeDistributed` Dense/softmax layer predicts one of 9 classes per frame: the 8 swing events plus a "no event" class.

See `Model.py` for the architecture.

## Repository layout

| File | Purpose |
|---|---|
| `Model.py` | CNN + (Bi)LSTM model definition |
| `GolfDataset.py` | Dataset/`tf.data` pipeline built from `GolfDB.pkl` and clip videos |
| `Train.py` | Training loop, callbacks (checkpointing, TensorBoard, LR schedule, CSV logging) |
| `Metrics.py` | Custom loss (`weighted_sparse_cce`) and metrics (event precision/recall/AUC) |
| `TestVideo.py` | Loads, resizes, and center-crops an arbitrary video for inference; runs frame-by-frame event predictions |
| `Utils.py` / `Visuals.py` | Helper and visualization utilities |
| `golfDatabasePickle.py` | Builds `GolfDB.pkl` from `GolfDB.csv` |
| `golfDatabaseDatasetVisualizationAndCleaning.py` | Exploration/cleaning of the GolfDB metadata |
| `GolfDB.csv` / `GolfDB.pkl` | GolfDB metadata (clip labels, event frame indices) |
| `scripts/download_data.py` | Fetches the preprocessed swing clips from the official GolfDB release |
| `Copy_of_GolfSwingTrainingAndValidation.ipynb` | Notebook version of the training/validation workflow |

## Requirements

- Python 3.9+ and TensorFlow / Keras

```bash
pip install -r requirements.txt
```

## Getting the data

The preprocessed `160x160` swing clips (`~680 MB`) are **not** committed to this
repository; they are downloaded on demand from the official GolfDB release.

```bash
python scripts/download_data.py
```

This downloads and extracts the clips into `videos_160/`. The `GolfDB.pkl`
metadata file (clip labels and event frame indices) is already included so
training can start immediately once the clips are present.

## Usage

### Train

```bash
python scripts/download_data.py   # first time only
python Train.py
```

This reads `GolfDB.pkl` and the clips in `videos_160/`, splits train/test, and trains the CNN-LSTM model. Checkpoints are saved to `models/`, per-run TensorBoard logs to `logs/<run_id>/`, and a CSV metric log to `logs/`.

### Run on a new video

```python
from tensorflow.keras.models import load_model
from TestVideo import TestVideo, compute_event_frames

model = load_model("path/to/model.keras")
video = TestVideo("path/to/swing.mp4")
event_frames = compute_event_frames(model, video.frames)
```

`TestVideo` resizes and center-crops the input video to `160x160` and preprocesses it for MobileNetV2. `compute_event_frames` returns the predicted frame index for each of the 8 swing events.

## Dataset

Trained on [GolfDB](https://arxiv.org/abs/1903.06528) (McNally et al., 2019), a video database of golf swings labeled with 8 event frames per clip.

## Status

Work in progress / research project — model architecture and training hyperparameters are still being iterated on.

## License

No license specified yet.
