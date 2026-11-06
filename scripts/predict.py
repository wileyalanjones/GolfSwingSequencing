"""Command-line interface for predicting swing events in a new video."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorflow.keras.models import load_model

from golfswing import metrics as mt
from golfswing.predict import TestVideo, compute_event_frames


def _load_model(path: str):
    """Load a saved .keras model, registering the custom loss and metrics."""
    return load_model(
        path,
        custom_objects={
            "weighted_sparse_cce": mt.weighted_sparse_cce,
            "EventPrecision": mt.EventPrecision,
            "EventRecall": mt.EventRecall,
            "EventAUC": mt.EventAUC,
        },
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict the 8 swing events in a video using a trained model."
    )
    parser.add_argument("video", help="Path to the input swing video.")
    parser.add_argument("model", help="Path to a saved .keras model.")
    args = parser.parse_args(argv)

    model = _load_model(args.model)
    video = TestVideo(args.video)
    event_frames = compute_event_frames(model, video.frames)

    print("Predicted event frames:", event_frames.tolist())


if __name__ == "__main__":
    main()