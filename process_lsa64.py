"""Step 4 (data source: LSA64): convert the downloaded LSA64 sign videos
into the same landmark-sequence format collect_data.py produces, so the
rest of the pipeline (explore_data.py, train.py, etc.) doesn't care whether
the data came from a webcam recording session or this public dataset.

LSA64 file naming: all_cut/<sign_id>_<subject_id>_<rep_id>.mp4
  sign_id:    001-064, which sign
  subject_id: 001-010, which person performed it
  rep_id:     001-005, which repetition

We use subject_id as the "session" for the leakage-safe split in step 6 —
splitting by subject is a stronger generalization test than splitting by
recording session anyway (the model has to work on a person it never saw
during training, not just a different sitting with the same person).

Run:
  ./venv/bin/python process_lsa64.py
"""
import csv
import os

import cv2
import numpy as np

from gestures import GESTURES
from hand_tracker import HandTracker

RAW_DIR = "lsa64_raw/all_cut"
DATA_DIR = "data"
SEQ_LEN = 30
MS_PER_FRAME = 34  # ~30fps equivalent, just needs to strictly increase

# LSA64 sign_id -> our gesture name, for the 10 signs we chose (see gestures.py)
SIGN_ID_TO_GESTURE = {
    "002": "red",
    "003": "green",
    "021": "milk",
    "022": "water",
    "023": "food",
    "039": "name",
    "051": "thanks",
    "056": "help",
    "059": "buy",
    "061": "run",
}


def extract_sequence(tracker, video_path, next_ts):
    """Runs every frame of the video through the hand tracker, keeping only
    frames where a hand was detected, then uniformly resamples to SEQ_LEN
    frames so all sequences end up the same length regardless of the
    source video's duration. Returns (sequence array or None, updated next_ts)."""
    cap = cv2.VideoCapture(video_path)
    landmarks_seq = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = tracker.process(frame, timestamp_ms=next_ts)
        next_ts += MS_PER_FRAME
        if raw is not None:
            landmarks_seq.append(raw)
    cap.release()

    if len(landmarks_seq) < 10:
        return None, next_ts  # too few detected frames to trust this clip

    landmarks_seq = np.stack(landmarks_seq)  # (n_frames, 21, 3)
    # uniform resampling to a fixed length
    indices = np.linspace(0, len(landmarks_seq) - 1, SEQ_LEN).astype(int)
    resampled = landmarks_seq[indices]
    return resampled, next_ts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    write_header = not os.path.exists(metadata_path)
    metadata_file = open(metadata_path, "a", newline="")
    writer = csv.writer(metadata_file)
    if write_header:
        writer.writerow(["filepath", "gesture", "session_id", "sequence_idx", "timestamp"])

    tracker = HandTracker()
    next_ts = 0

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".mp4"))
    print(f"Processing {len(files)} videos...")

    counts = {g: 0 for g in GESTURES}
    skipped = 0

    for i, fname in enumerate(files):
        sign_id, subject_id, rep_id = fname[:-4].split("_")
        gesture = SIGN_ID_TO_GESTURE.get(sign_id)
        if gesture is None:
            continue  # not one of our chosen 10 signs (shouldn't happen, we pre-filtered on extract)

        video_path = os.path.join(RAW_DIR, fname)
        seq, next_ts = extract_sequence(tracker, video_path, next_ts)

        if seq is None:
            skipped += 1
            print(f"  [{i+1}/{len(files)}] {fname}: skipped (too few frames with a detected hand)")
            continue

        gesture_dir = os.path.join(DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        session_id = f"subject_{subject_id}"
        out_name = f"{session_id}_{rep_id}.npy"
        out_path = os.path.join(gesture_dir, out_name)
        np.save(out_path, seq)

        writer.writerow([out_path, gesture, session_id, counts[gesture], fname])
        metadata_file.flush()
        counts[gesture] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] processed...")

    tracker.close()
    metadata_file.close()

    print("\nDone.")
    for g in GESTURES:
        print(f"  {g}: {counts[g]} sequences")
    print(f"  skipped (no hand detected): {skipped}")


if __name__ == "__main__":
    main()
