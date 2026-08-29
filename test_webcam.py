"""Step 1 sanity check: confirm OpenCV can open the webcam and grab a frame."""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam at index 0")

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Webcam opened but failed to read a frame")

print("Webcam OK. Frame shape:", frame.shape)
cv2.imwrite("webcam_test.jpg", frame)
print("Saved webcam_test.jpg")
