# Golf Swing Sequencing

A deep learning model that detects the 8 key events of a golf swing (address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish) directly from raw swing video, trained on the [GolfDB](https://github.com/wmcnally/golfdb) dataset. Includes a web app: upload a swing video and get the 8 event frames back, which you can enlarge and download.

<!-- TODO: add a screenshot or GIF of the web app, e.g. docs/demo.gif -->
![Web app demo](docs/demo.gif)

**Live demo:** <!-- TODO: add link once deployed -->

## How it works

The model treats swing-event detection as per-frame classification over a video clip:

- **CNN backbone** — a `MobileNetV2` (ImageNet weights) extracts features from each `160x160` frame.
- **Temporal head** — a `TimeDistributed` wrapper applies the CNN to every frame in a clip, then a bidirectional LSTM (256 units) models motion across time.
- **Output** — a `TimeDistributed` Dense layer predicts one of 9 classes per frame: the 8 swing events plus a "no event" class.
- **Loss** — weighted sparse cross-entropy, since event frames are rare compared to background frames.

Clips are 64 frames long. At inference time, the video is split into clips, the per-frame scores are combined, and the 8 event frames are picked in order (an event can't come before the one preceding it).

### Training

Training runs in two stages (see `scripts/train.py`):

1. **Head only** — the CNN is frozen and only the LSTM head trains (`lr=1e-3`).
2. **Fine-tune** — the last 30 CNN layers are unfrozen (BatchNorm layers stay frozen) and everything trains at a lower learning rate (`lr=1e-4`).

The best checkpoint is chosen by **PCE** (Percentage of Correct Events) on whole validation videos, not by frame-level metrics. Learning rate reduction and early stopping also follow validation PCE.

### Results

<!-- TODO: fill in with your numbers from the final model -->

| Metric | Value |
| ------ | ----- |
| PCE (test set) | _TODO_ |

See `golfswing/model.py` for the architecture and `golfswing/metrics.py` for the loss and PCE implementation.

## Web app

The web app has a React + TypeScript + Tailwind frontend (`frontend/`) and a FastAPI backend (`webapp/`) that loads the trained model and exposes:

- `POST /api/analyze` — accepts a video upload (`.mp4`, `.mov`, `.m4v`, `.avi`, up to 100 MB) and returns the 8 event frames as JPEGs with their frame indices.
- `GET /health` — health check.

### Run it

Requires Python 3.12 and Node.js. You also need a trained weights file at the path set by `MODEL_PATH` in `webapp/main.py` (default `models/20261108-025410_best.weights.h5`). The `models/` folder is not committed.

```bash
# 1. Install the Python package and web dependencies
pip install -e ".[web]"

# 2. Build the frontend (once, and after any frontend change)
cd frontend
npm install
npm run build
cd ..

# 3. Start the server from the repo root
python -m uvicorn webapp.main:app
```

Then open http://127.0.0.1:8000.

### Develop the frontend

For live reloading, run the backend and the Vite dev server side by side. Vite proxies `/api` requests to the backend.

```bash
# terminal 1 (repo root)
python -m uvicorn webapp.main:app --reload

# terminal 2
cd frontend && npm run dev
```

Open http://localhost:5173.

## Repository layout

```
golfswing/        # the library
  ├── model.py    # CNN + (Bi)LSTM model definition
  ├── dataset.py  # Dataset / tf.data pipeline built from GolfDB.pkl and clips
  ├── metrics.py  # Custom loss, event metrics, and PCE
  ├── callbacks.py# Validation-PCE callback used during training
  ├── predict.py  # Video loading, cropping, and event-frame prediction
  ├── utils.py    # Frame/event helper utilities
  ├── visualize.py# Training-metric and PCE plotting utilities
  └── data.py     # GolfDB loading and CSV -> pickle conversion

webapp/           # FastAPI backend (serves the built frontend)
frontend/         # React + TypeScript + Tailwind frontend (Vite)

scripts/
  ├── train.py           # training entrypoint
  ├── predict.py         # CLI for predicting events in a new video
  └── download_data.py   # fetches the swing clips from GolfDB

data/
  ├── GolfDB.csv         # GolfDB metadata (tracked)
  └── GolfDB.pkl         # pickled metadata used for training (tracked)
```

## Requirements

- Python 3.9+ (3.12 recommended) and TensorFlow / Keras
- Node.js (only to build the web frontend)

```bash
pip install -e .          # installs the package + dependencies
pip install -e ".[web]"   # also installs the web app dependencies
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
train/test, and trains the CNN-LSTM model in two stages. Checkpoints are saved to
`models/`, per-run TensorBoard logs to `logs/<run_id>/`, and a CSV metric log to
`logs/`.

### Predict on a new video

```bash
python scripts/predict.py path/to/swing.mp4 path/to/model.keras
```

`predict.py` resizes and center-crops the input video to `160x160`, preprocesses
it for MobileNetV2, and prints the predicted frame index for each of the 8 swing
events.

## Dataset

Trained on [GolfDB](https://arxiv.org/abs/1903.06528) (McNally et al., 2019), a video database of golf swings labeled with 8 event frames per clip.

## License

This project's code is released under the [MIT License](LICENSE).

The GolfDB dataset is © the original authors and is provided under a
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
license — see the [GolfDB repository](https://github.com/wmcnally/golfdb) for details.
