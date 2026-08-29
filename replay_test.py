"""Test (and demo) the live-inference pipeline without needing to know any
sign language yourself: replays real held-out test videos through the
exact same sliding-window + model code path as live_demo.py, instead of
a live webcam feed. Also saves annotated output videos you can use as the
demo clip in your README (step 11), since you can't record yourself
performing LSA64 signs you don't know.

Run:
  ./venv/bin/python replay_test.py [--n 6] [--seed 42]
"""
import argparse
import csv
import os
import random
from collections import deque

import cv2
import numpy as np
import torch

from gestures import GESTURES
from hand_tracker import HandTracker, draw_landmarks, normalize_landmarks
from model import GestureLSTM

SEQ_LEN = 30
MS_PER_FRAME = 34
RAW_DIR = "lsa64_raw/all_cut"
OUT_DIR = "demo_output"


def load_test_clips():
    # metadata.csv's last column holds the source LSA64 mp4 filename;
    # splits.csv holds the train/val/test assignment. Join on filepath.
    source_video = {}
    with open("data/metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            source_video[row["filepath"]] = row["timestamp"]

    clips = []
    with open("data/splits.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            src = source_video.get(row["filepath"])
            if src:
                clips.append((row["gesture"], src))
    return clips


def run_on_video(model, tracker, video_path, next_ts):
    """next_ts must be a counter shared across the tracker's whole lifetime
    (not reset per video) -- mediapipe requires strictly increasing
    timestamps across every call to the same landmarker instance, not just
    within one clip. Returns (frames_out, updated next_ts)."""
    cap = cv2.VideoCapture(video_path)
    buffer = deque(maxlen=SEQ_LEN)
    frames_out = []
    pred_label, pred_conf = "-", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = tracker.process(frame, timestamp_ms=next_ts)
        next_ts += MS_PER_FRAME
        display = frame.copy()

        if raw is not None:
            display = draw_landmarks(display, raw)
            buffer.append(normalize_landmarks(raw))
        else:
            buffer.clear()  # same policy as live_demo.py

        if len(buffer) == SEQ_LEN:
            x = torch.from_numpy(np.stack(buffer, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
            idx = int(probs.argmax())
            pred_label, pred_conf = GESTURES[idx], float(probs[idx])

        frames_out.append((display, pred_label, pred_conf))

    cap.release()
    return frames_out, next_ts


def save_annotated_video(frames_out, true_label, out_path):
    if not frames_out:
        return
    h, w = frames_out[0][0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for frame, pred_label, pred_conf in frames_out:
        color = (0, 200, 0) if pred_label == true_label else (0, 0, 255)
        text = f"true: {true_label} | pred: {pred_label} ({pred_conf:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clips = load_test_clips()
    rng = random.Random(args.seed)
    rng.shuffle(clips)
    chosen = clips[: args.n]

    model = GestureLSTM()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    tracker = HandTracker()

    os.makedirs(OUT_DIR, exist_ok=True)
    correct = 0
    next_ts = 0

    print(f"Replaying {len(chosen)} held-out test clips through the live-inference pipeline...\n")
    for gesture, src in chosen:
        video_path = os.path.join(RAW_DIR, src)
        frames_out, next_ts = run_on_video(model, tracker, video_path, next_ts)
        final_pred = frames_out[-1][1] if frames_out else "-"
        ok = final_pred == gesture
        correct += ok

        status = "OK" if ok else "WRONG"
        print(f"  true={gesture:<8} final_pred={final_pred:<8} {status}  ({src})")

        out_name = f"{gesture}_{src.replace('.mp4', '')}_annotated.mp4"
        save_annotated_video(frames_out, gesture, os.path.join(OUT_DIR, out_name))

    tracker.close()
    print(f"\n{correct}/{len(chosen)} correct on this replay sample")
    print(f"Annotated demo videos saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
