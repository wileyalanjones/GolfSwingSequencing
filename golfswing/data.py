"""Utilities for loading and preparing the GolfDB dataset.

The raw GolfDB release ships a ``GolfDB.csv`` with list/array-valued columns
stored as strings. The training code consumes a pickled DataFrame
(``GolfDB.pkl``) where those columns are real Python objects. This module wraps
both the loading and the (one-time) conversion from CSV to pickle.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PKL_PATH = DATA_DIR / "GolfDB.pkl"


def load_golfdb(pkl_path=None) -> pd.DataFrame:
    """Load the GolfDB metadata as a pandas DataFrame."""
    pkl_path = Path(pkl_path) if pkl_path is not None else DEFAULT_PKL_PATH
    return pd.read_pickle(pkl_path)


def split_golfdb(dataframe, test_size=0.25, seed=42):
    """Split GolfDB into train/test the same way every time.

    Use this everywhere (training, evaluation) so every model is scored
    on the same held-out videos.
    """
    return train_test_split(dataframe, test_size=test_size, random_state=seed)


def _parse_int_list(value) -> list:
    """Parse a "[0, 47, 65, ...]" string into a list of ints."""
    return ast.literal_eval(value)


def _parse_float_array(value) -> np.ndarray:
    """Parse a "[0.09 0.0 0.5 0.98]" string into a float array."""
    return np.array([float(x) for x in str(value).strip("[]").split()])


def build_pickle(csv_path, pkl_path=None) -> pd.DataFrame:
    """Build ``GolfDB.pkl`` from the raw ``GolfDB.csv``.

    Converts the string-encoded ``events`` and ``bbox`` columns into the
    Python objects expected by ``golfswing.dataset``.
    """
    csv_path = Path(csv_path)
    pkl_path = Path(pkl_path) if pkl_path is not None else csv_path.with_suffix(".pkl")

    df = pd.read_csv(csv_path, index_col=0)
    df["events"] = df["events"].apply(_parse_int_list)
    df["bbox"] = df["bbox"].apply(_parse_float_array) # type: ignore
    df.to_pickle(pkl_path)

    return df