"""Loads a split (train/val/test) from data/splits.csv into normalized
numpy arrays ready for training. Separated from split_data.py so re-running
the split doesn't require re-deriving this loading logic each time.
"""
import csv

import numpy as np

from gestures import GESTURES
from hand_tracker import normalize_landmarks

SPLITS_PATH = "data/splits.csv"
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}


def load_split(split_name):
    """Returns (X, y): X is (n_sequences, seq_len, 63) normalized landmark
    features, y is (n_sequences,) integer gesture labels (index into
    GESTURES)."""
    X, y = [], []
    with open(SPLITS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split_name:
                continue
            raw = np.load(row["filepath"])  # (seq_len, 21, 3)
            normalized = np.stack([normalize_landmarks(frame) for frame in raw])  # (seq_len, 63)
            X.append(normalized)
            y.append(GESTURE_TO_IDX[row["gesture"]])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        X, y = load_split(split)
        print(f"{split}: X={X.shape} y={y.shape}")
