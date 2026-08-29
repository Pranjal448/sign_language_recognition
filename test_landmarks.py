"""Step 3 sanity check: capture a frame, run hand landmark detection,
draw the skeleton, and print the normalized feature vector shape."""
import cv2

from hand_tracker import HandTracker, normalize_landmarks, draw_landmarks

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam at index 0")

tracker = HandTracker()

# grab a few frames — first frame from a webcam is sometimes dark/stale
frame = None
for _ in range(5):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read a frame")
cap.release()

raw = tracker.process(frame)
tracker.close()

if raw is None:
    print("No hand detected in this frame. Saving raw frame for inspection.")
    cv2.imwrite("landmarks_test.jpg", frame)
else:
    print("Hand detected. Raw landmarks shape:", raw.shape)
    normalized = normalize_landmarks(raw)
    print("Normalized feature vector shape:", normalized.shape)
    annotated = draw_landmarks(frame, raw)
    cv2.imwrite("landmarks_test.jpg", annotated)
    print("Saved landmarks_test.jpg")
