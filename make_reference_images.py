"""Saves one labeled reference frame per gesture into sign_language/, so
you have a real hand-shape picture to look at (and mimic) for each sign,
without needing to know LSA64 sign language yourself.

Run:
  ./venv/bin/python make_reference_images.py
"""
import os

import cv2

from gestures import GESTURES

RAW_DIR = "lsa64_raw/all_cut"
OUT_DIR = "sign_language"

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
GESTURE_TO_SIGN_ID = {v: k for k, v in SIGN_ID_TO_GESTURE.items()}


def middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for gesture in GESTURES:
        sign_id = GESTURE_TO_SIGN_ID[gesture]
        # subject_001, repetition 1
        video_path = os.path.join(RAW_DIR, f"{sign_id}_001_001.mp4")
        frame = middle_frame(video_path)
        if frame is None:
            print(f"  {gesture}: could not read {video_path}")
            continue

        label = gesture.upper()
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out_path = os.path.join(OUT_DIR, f"{gesture}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
