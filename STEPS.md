# Sign Language / Hand Gesture Recognition — Project Steps

Goal: recognize a set of hand gestures (e.g. ASL alphabet or a custom set of
10-15 commands) live from a webcam, using MediaPipe hand landmarks + an LSTM
for temporal sequence classification.

## 1. Set up the environment
Create a virtualenv, install `opencv-python`, `mediapipe`, `numpy`,
`torch` (or `tensorflow`), `scikit-learn`, `matplotlib`. Verify your webcam
works with a minimal OpenCV script that just shows the camera feed.

## 2. Pick your gesture set
Decide the list of gestures you'll classify (start small: 8-12 classes is
plenty for a first version — e.g. a subset of the ASL alphabet, or custom
commands like "stop", "go", "yes", "no", "thumbs up"). Write them down —
this becomes your label list.

## 3. Build the landmark extraction pipeline
Write a script that reads webcam frames, runs MediaPipe Hands on each frame,
and extracts the 21 (x, y, z) landmark points per detected hand. Normalize
the coordinates (e.g. relative to the wrist point, scaled by hand bounding
box) so the model doesn't just learn hand position/size instead of shape.

## 4. Collect your dataset
For each gesture, record multiple short sequences (e.g. 30 frames each) of
yourself performing it, varying lighting, hand angle, and distance from the
camera a bit. Save each sequence as a `.npy` array of landmark coordinates
plus its label. Aim for at least 30-50 sequences per gesture to start.
**Track which recording "session" each sequence came from** — you'll need
this in step 6.

## 5. Explore and sanity-check the data
Plot a few landmark sequences, check class balance (roughly equal samples
per gesture), and visually confirm a few samples per gesture actually look
like the gesture (bad or mislabeled data is the #1 cause of a model that
won't learn).

## 6. Split into train/val/test — by session, not by frame or sequence
Hold out entire recording sessions for validation/test, never split
individual sequences from the same session across sets. This prevents data
leakage (the model memorizing your specific session rather than the
gesture).

## 7. Build the LSTM model
A small model: input = sequence of landmark vectors (e.g. 30 frames × 63
values for one hand), 1-2 LSTM layers, followed by a fully connected layer
+ softmax over your gesture classes. Keep it small — this is a tiny-input
problem, it doesn't need to be deep.

## 8. Train the model
Standard supervised training loop: cross-entropy loss, Adam optimizer,
track train vs. validation loss/accuracy per epoch. Watch for overfitting
(val loss rising while train loss falls) — if it happens, that's a real
learning moment, not a failure (try more data, dropout, or a smaller
model).

## 9. Evaluate properly
Run on the held-out test set. Report accuracy, and a confusion matrix per
gesture (this tells you *which* gestures get confused with each other,
which is far more useful than a single accuracy number and a good thing to
show in an interview).

## 10. Build the live inference demo
Wire the trained model back into the webcam pipeline: continuously extract
landmarks from a sliding window of recent frames, feed them to the model,
and display the predicted gesture on-screen in real time.

## 11. Write up the project
README with: problem statement, pipeline diagram, dataset description,
model architecture, results (accuracy + confusion matrix), and a short
demo GIF/video. This is what you'll actually show in interviews and on
your resume.

## 12. (Optional stretch) Improve robustness
Try: more gestures, more data diversity (different people, backgrounds),
data augmentation (small rotations/translations of landmarks), or comparing
LSTM vs. GRU vs. a simple 1D-CNN baseline to see what you learn about the
tradeoffs.
