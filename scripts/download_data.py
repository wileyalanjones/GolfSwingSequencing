#!/usr/bin/env python3
"""
Download the GolfDB preprocessed swing videos and place them where the
training code expects them.

The videos are not committed to this repository (they are ~680 MB of 160x160
MP4 clips) and are instead fetched from the official GolfDB release:

    McNally et al., "GolfDB: A Video Database for Golf Swing Sequencing",
    CVPR Workshops, 2019.  https://github.com/wmcnally/golfdb

Usage:
    python scripts/download_data.py
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Official GolfDB 160x160 clip archive (see wmcnally/golfdb README).
GOLFDB_VIDEOS_URL = "https://drive.google.com/uc?id=1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos_160"
ZIP_PATH = DATA_DIR / "videos_160.zip"


def download_with_gdown(url: str, dest: Path) -> None:
    """Download a Google Drive file using `gdown` (handles large-file tokens)."""
    try:
        import gdown  # type: ignore
    except ImportError:
        print("The `gdown` package is required to download the dataset.")
        print("Install it with:  pip install gdown")
        sys.exit(1)

    print(f"Downloading GolfDB videos to {dest} ...")
    gdown.download(url, str(dest), quiet=False, fuzzy=True)
    if not dest.exists():
        print(f"Download failed: {dest} was not created.")
        sys.exit(1)


def download_with_curl(url: str, dest: Path) -> None:
    """Fallback: stream the file with curl (follows Google's confirm redirect)."""
    print(f"Downloading GolfDB videos to {dest} (curl) ...")
    subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), url], check=True)
    if not dest.exists():  # pragma: no cover
        print(f"Download failed: {dest} was not created.")
        sys.exit(1)


def extract_zip(zip_path: Path) -> None:
    """Extract videos_160.zip and make sure clips live in data/videos_160/."""
    print(f"Extracting {zip_path} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # Some archives wrap everything in a nested `videos_160/` folder.
    nested = VIDEOS_DIR / "videos_160"
    if nested.is_dir():
        shutil.copytree(nested, VIDEOS_DIR, dirs_exist_ok=True)
        shutil.rmtree(nested)

    n_mp4 = len(list(VIDEOS_DIR.glob("*.mp4"))) if VIDEOS_DIR.exists() else 0
    print(f"Done. Found {n_mp4} clips in {VIDEOS_DIR}.")


def main() -> None:
    if VIDEOS_DIR.exists() and list(VIDEOS_DIR.glob("*.mp4")):
        print(f"{VIDEOS_DIR} already contains clips; nothing to do.")
        return

    if ZIP_PATH.exists():
        print(f"Found existing {ZIP_PATH}; skipping download.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            download_with_gdown(GOLFDB_VIDEOS_URL, ZIP_PATH)
        except Exception:
            print("gdown failed; falling back to curl.")
            download_with_curl(GOLFDB_VIDEOS_URL, ZIP_PATH)

    extract_zip(ZIP_PATH)
    # Keep the archive out of git; clean it up after extraction.
    ZIP_PATH.unlink(missing_ok=True)
    print("Removed temporary archive.")


if __name__ == "__main__":
    main()