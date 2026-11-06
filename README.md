# Golf Swing Sequencing

A deep learning model that detects the 8 key events of a golf swing (address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish) directly from raw swing video, trained on the [GolfDB](https://github.com/wmcnally/golfdb) dataset.

## How it works

The model treats swing-event detection as per-frame classification over a video clip:

- **CNN backbone** — a frozen `MobileNetV2` (ImageNet weights) extracts features from each `160x160` frame.
- **Temporal head** — a `TimeDistributed` wrapper applies the CNN to every frame in a clip, then a (bidirectional) LSTM models motion across time.
- **Output** — a `TimeDistributed` Dense/softmax layer predicts one of 9 classes per frame: the 8 swing events plus a "no event" class.

See `golfswing/model.py` for the architecture.

## Repository layout

```
golfswing/        # the library
  ├── model.py    # CNN + (Bi)LSTM model definition
  ├── dataset.py  # Dataset / tf.data pipeline built from GolfDB.pkl and clips
  ├── metrics.py  # Custom loss and event precision/recall/AUC metrics
  ├── predict.py  # Video loading, cropping, and event-frame prediction
  ├── utils.py    # Frame/event helper utilities
  ├── visualize.py# Training-metric and PCE plotting utilities
  └── data.py     # GolfDB loading and CSV -> pickle conversion

scripts/
  ├── train.py           # training entrypoint
  ├── predict.py         # CLI for predicting events in a new video
  └── download_data.py   # fetches the swing clips from GolfDB

notebooks/
  └── training_and_validation.ipynb   # original Colab experiment notebook

data/
  ├── GolfDB.csv         # GolfDB metadata (tracked)
  └── GolfDB.pkl         # pickled metadata used for training (tracked)

results/
  └── *.csv              # example training-metric logs used by visualize.py
```

## Requirements

- Python 3.9+ and TensorFlow / Keras

```bash
pip install -e .          # installs the package + dependencies
```

Alternatively, `pip install -r requirements.txt` installs just the
dependencies without the package itself.

## Getting the data

The preprocessed `160x160` swing clips (`~680 MB`) are **not** committed to this
repository; they are downloaded on demand from the official GolfDB release.

```bash
python scripts/download_data.py
```

This downloads and extracts the clips into `data/videos_160/`. The `GolfDB.pkl`
metadata file (clip labels and event frame indices) is already included so
training can start immediately once the clips are present.

## Usage

### Train

```bash
python scripts/download_data.py   # first time only
python scripts/train.py
```

This reads `data/GolfDB.pkl` and the clips in `data/videos_160/`, splits
train/test, and trains the CNN-LSTM model. Checkpoints are saved to `models/`,
per-run TensorBoard logs to `logs/<run_id>/`, and a CSV metric log to `logs/`.

### Predict on a new video

```bash
python scripts/predict.py path/to/swing.mp4 path/to/model.keras
```

`predict.py` resizes and center-crops the input video to `160x160`, preprocesses
it for MobileNetV2, and prints the predicted frame index for each of the 8 swing
events.

## Dataset

Trained on [GolfDB](https://arxiv.org/abs/1903.06528) (McNally et al., 2019), a video database of golf swings labeled with 8 event frames per clip.

## Status

Work in progress / research project — model architecture and training hyperparameters are still being iterated on.

## License

This project's code is released under the [MIT License](LICENSE).

The GolfDB dataset is © the original authors and is provided under a
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
license — see the [GolfDB repository](https://github.com/wmcnally/golfdb) for details.