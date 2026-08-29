"""Step 10: live webcam demo.

Continuously extracts hand landmarks, keeps a sliding window of the last
SEQ_LEN frames, and runs the trained LSTM on that window every frame so
the prediction updates live as you sign.

Caveat worth knowing: the model was trained on LSA64 clips that were each
resampled to exactly 30 frames spanning one full sign performance. Here,
the sliding window is just "the last 30 webcam frames," whose real-time
duration depends on your camera's frame rate and how fast MediaPipe runs
on this machine -- so for best results, perform each sign at a similar
pace to a natural, one-to-two-second gesture, and expect it to feel less
reliable than the offline test accuracy. This mismatch between training-
time and inference-time framing is a common, worth-mentioning limitation
of sliding-window sequence models.

Controls: Q to quit.
Run:
  ./venv/bin/python live_demo.py
"""
from collections import deque

import cv2
import numpy as np
import torch

from gestures import GESTURES
from hand_tracker import HandTracker, draw_landmarks, normalize_landmarks
from model import GestureLSTM

SEQ_LEN = 30
CONF_THRESHOLD = 0.6


def main():
    model = GestureLSTM()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam at index 0")

    buffer = deque(maxlen=SEQ_LEN)
    pred_label, pred_conf = "-", 0.0

    print("Live demo running. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        raw = tracker.process(frame)
        display = frame.copy()

        if raw is not None:
            display = draw_landmarks(display, raw)
            buffer.append(normalize_landmarks(raw))
        else:
            buffer.clear()  # require a fresh, unbroken window once the hand reappears

        if len(buffer) == SEQ_LEN:
            x = torch.from_numpy(np.stack(buffer, dtype=np.float32)).unsqueeze(0)  # (1, seq_len, 63)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
            idx = int(probs.argmax())
            pred_label, pred_conf = GESTURES[idx], float(probs[idx])

        if pred_conf >= CONF_THRESHOLD:
            text, color = f"{pred_label} ({pred_conf:.2f})", (0, 200, 0)
        else:
            text, color = f"...({pred_label} {pred_conf:.2f})", (0, 165, 255)

        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Sign Language Recognition - live demo", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
