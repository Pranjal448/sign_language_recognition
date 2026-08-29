"""Step 4: data collection.

Records short landmark sequences for each gesture in gestures.py and saves
them to data/<gesture>/<session_id>_<index>.npy, with a metadata.csv row
per sequence tracking which recording session it came from (needed for the
leakage-safe split in step 6).

Controls (shown on-screen):
  SPACE - start recording one sequence for the current gesture
  N     - move on to the next gesture
  Q     - quit

Run from the project directory:
  ./venv/bin/python collect_data.py [--samples-per-gesture 40] [--seq-len 30]
"""
import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np

from gestures import GESTURES
from hand_tracker import HandTracker, draw_landmarks

DATA_DIR = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-gesture", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DATA_DIR, exist_ok=True)

    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    write_header = not os.path.exists(metadata_path)
    metadata_file = open(metadata_path, "a", newline="")
    writer = csv.writer(metadata_file)
    if write_header:
        writer.writerow(["filepath", "gesture", "session_id", "sequence_idx", "timestamp"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam at index 0")

    tracker = HandTracker()

    print(f"Session ID: {session_id}")
    print(f"Target: {args.samples_per_gesture} sequences per gesture, {args.seq_len} frames each")
    print("Controls: SPACE = record one sequence | N = next gesture | Q = quit\n")

    quit_all = False

    for gesture in GESTURES:
        if quit_all:
            break

        gesture_dir = os.path.join(DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        existing = len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
        collected_this_session = 0

        print(f"=== Gesture: {gesture} ===  (already have {existing} samples)")

        recording = False
        buffer = []

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            raw = tracker.process(frame)
            display = frame.copy()
            if raw is not None:
                display = draw_landmarks(display, raw)

            if recording:
                text = f"RECORDING '{gesture}'  {len(buffer)}/{args.seq_len}"
                color = (0, 0, 255)
                if raw is not None:
                    buffer.append(raw)
                # if no hand detected mid-recording, we simply don't append
                # a frame this tick and keep waiting for the hand to reappear
                if len(buffer) >= args.seq_len:
                    seq_arr = np.stack(buffer[: args.seq_len])
                    idx = existing + collected_this_session
                    fname = f"{session_id}_{idx:03d}.npy"
                    fpath = os.path.join(gesture_dir, fname)
                    np.save(fpath, seq_arr)
                    writer.writerow([fpath, gesture, session_id, idx, datetime.now().isoformat()])
                    metadata_file.flush()
                    collected_this_session += 1
                    recording = False
                    buffer = []
                    print(f"  saved sample {idx} ({collected_this_session}/{args.samples_per_gesture} this session)")
            else:
                text = (f"'{gesture}': {collected_this_session}/{args.samples_per_gesture} this session "
                        f"(total {existing + collected_this_session})  |  SPACE=record  N=next  Q=quit")
                color = (0, 255, 0)

            cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow("Data Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and not recording:
                recording = True
                buffer = []
            elif key == ord("n"):
                break
            elif key == ord("q"):
                quit_all = True
                break
            elif collected_this_session >= args.samples_per_gesture and not recording:
                # auto-advance once target is hit; N/Q still work above
                break

        print()

    cap.release()
    tracker.close()
    metadata_file.close()
    cv2.destroyAllWindows()
    print("Done. Data saved under data/, metadata in data/metadata.csv")


if __name__ == "__main__":
    main()
