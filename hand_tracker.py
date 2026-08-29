"""Step 3: hand landmark extraction pipeline.

Wraps MediaPipe's HandLandmarker (Tasks API) so the rest of the project
just deals with plain numpy arrays, not MediaPipe's own types.
"""
import time

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 21
MODEL_PATH = "hand_landmarker.task"

# Landmark connections for drawing (pairs of landmark indices), matches
# MediaPipe's standard hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


class HandTracker:
    """Extracts one hand's 21 landmarks from BGR frames (as from OpenCV)."""

    def __init__(self, model_path=MODEL_PATH, num_hands=1):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

    def _timestamp_ms(self):
        return int((time.monotonic() - self._start_time) * 1000)

    def process(self, frame_bgr, timestamp_ms=None):
        """Returns raw landmarks as an (21, 3) array of (x, y, z) in
        MediaPipe's normalized image coordinates, or None if no hand
        was detected.

        timestamp_ms: pass an explicit, strictly-increasing timestamp when
        processing pre-recorded video frames in a tight loop (wall-clock
        time can repeat within the same millisecond and mediapipe requires
        strictly increasing timestamps in VIDEO mode). Left as None for
        live webcam use, which times itself.
        """
        if timestamp_ms is None:
            timestamp_ms = self._timestamp_ms()
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]  # first detected hand
        return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    def close(self):
        self._landmarker.close()


def normalize_landmarks(raw):
    """Makes a (21, 3) raw landmark array translation- and scale-invariant:
    - subtract the wrist (landmark 0) so position in frame doesn't matter
    - divide by the max distance from wrist to any landmark, so hand size
      / distance from camera doesn't matter
    Returns a flattened (63,) vector.
    """
    wrist = raw[0]
    centered = raw - wrist
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-6:
        scale = 1e-6
    normalized = centered / scale
    return normalized.flatten()


def draw_landmarks(frame_bgr, raw_landmarks, color=(0, 255, 0)):
    """Draws landmarks + skeleton connections onto a copy of the frame for
    visual debugging. raw_landmarks are in normalized [0,1] image coords."""
    import cv2

    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in raw_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)

    return frame
