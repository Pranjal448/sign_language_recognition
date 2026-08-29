"""Step 5: explore and sanity-check the collected data.

Run after collect_data.py has produced at least a few samples:
  ./venv/bin/python explore_data.py

Produces:
  - console report: per-gesture / per-session sample counts, shape checks
  - data_class_balance.png: bar chart of samples per gesture
  - data_samples_grid.png: first frame of a few sample sequences per
    gesture, rendered as hand skeletons, so you can visually confirm the
    recordings actually look like the intended gesture
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestures import GESTURES
from hand_tracker import draw_landmarks

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
CANVAS_SIZE = (480, 640, 3)  # matches webcam frame shape used at collection time
SAMPLES_PER_GESTURE_IN_GRID = 3


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"No {METADATA_PATH} found. Run collect_data.py first."
        )
    rows = []
    with open(METADATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_metadata()
    if not rows:
        print("metadata.csv exists but is empty. Collect some data first.")
        return

    by_gesture = defaultdict(list)
    by_session = defaultdict(set)
    for row in rows:
        by_gesture[row["gesture"]].append(row)
        by_session[row["gesture"]].add(row["session_id"])

    # --- console report ---
    print(f"Total sequences: {len(rows)}")
    print(f"{'gesture':<12} {'count':>6} {'sessions':>9}")
    for g in GESTURES:
        count = len(by_gesture.get(g, []))
        n_sessions = len(by_session.get(g, []))
        flag = ""
        if count == 0:
            flag = "  <-- MISSING"
        elif n_sessions < 2:
            flag = "  <-- only 1 session (can't split by session yet)"
        print(f"{g:<12} {count:>6} {n_sessions:>9}{flag}")

    missing = [g for g in GESTURES if len(by_gesture.get(g, [])) == 0]
    if missing:
        print(f"\nWARNING: no data yet for: {', '.join(missing)}")

    counts = [len(by_gesture.get(g, [])) for g in GESTURES]
    if counts:
        imbalance = (max(counts) - min(counts)) / max(1, max(counts))
        if imbalance > 0.3:
            print(f"\nWARNING: class imbalance is high ({imbalance:.0%} spread) — "
                  "consider recording more for underrepresented gestures.")

    # --- shape / integrity check ---
    bad_files = []
    for row in rows:
        path = row["filepath"]
        if not os.path.exists(path):
            bad_files.append((path, "file missing"))
            continue
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[1:] != (21, 3):
            bad_files.append((path, f"unexpected shape {arr.shape}"))
    if bad_files:
        print(f"\n{len(bad_files)} problem file(s):")
        for path, reason in bad_files[:10]:
            print(f"  {path}: {reason}")
    else:
        print("\nAll sequence files have consistent shape (seq_len, 21, 3).")

    # --- class balance chart ---
    plt.figure(figsize=(8, 4))
    plt.bar(GESTURES, counts)
    plt.ylabel("sequences")
    plt.title("Samples per gesture")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("data_class_balance.png")
    plt.close()
    print("\nSaved data_class_balance.png")

    # --- visual sample grid: first frame of a few sequences per gesture ---
    present_gestures = [g for g in GESTURES if by_gesture.get(g)]
    if present_gestures:
        n_cols = SAMPLES_PER_GESTURE_IN_GRID
        n_rows = len(present_gestures)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]

        for r, gesture in enumerate(present_gestures):
            samples = by_gesture[gesture][:n_cols]
            for c in range(n_cols):
                ax = axes[r][c] if n_cols > 1 else axes[r]
                ax.axis("off")
                if c < len(samples):
                    arr = np.load(samples[c]["filepath"])
                    canvas = np.full(CANVAS_SIZE, 255, dtype=np.uint8)
                    frame = draw_landmarks(canvas, arr[0])

                    # crop tightly around the hand (with padding) so it's
                    # actually visible instead of a speck on a big canvas
                    h, w = CANVAS_SIZE[:2]
                    xs = arr[0][:, 0] * w
                    ys = arr[0][:, 1] * h
                    pad = 40
                    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
                    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
                    frame = frame[y0:y1, x0:x1]

                    ax.imshow(frame[:, :, ::-1])  # BGR -> RGB for matplotlib
                if c == 0:
                    ax.set_ylabel(gesture)
                    ax.axis("on")
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.tight_layout()
        plt.savefig("data_samples_grid.png", dpi=100)
        plt.close()
        print("Saved data_samples_grid.png — check that each row visually matches its gesture label")


if __name__ == "__main__":
    main()
