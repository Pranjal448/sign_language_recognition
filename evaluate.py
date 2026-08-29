"""Step 9: evaluate on the held-out test set (subjects never seen during
training or validation) and report a confusion matrix — which tells you
*which* gestures get confused with each other, far more useful for
debugging (and for an interview) than a single accuracy number.

Run:
  ./venv/bin/python evaluate.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from dataset import load_split
from gestures import GESTURES
from model import GestureLSTM


def main():
    X_test, y_test = load_split("test")
    print(f"test: {X_test.shape}")

    model = GestureLSTM()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = logits.argmax(dim=1).numpy()

    acc = (preds == y_test).mean()
    print(f"\nTest accuracy: {acc:.3f} ({(preds == y_test).sum()}/{len(y_test)})")

    print("\nPer-class report:")
    print(classification_report(y_test, preds, target_names=GESTURES, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=range(len(GESTURES)))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(GESTURES)))
    ax.set_yticks(range(len(GESTURES)))
    ax.set_xticklabels(GESTURES, rotation=45, ha="right")
    ax.set_yticklabels(GESTURES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (test acc={acc:.2f})")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nSaved confusion_matrix.png")


if __name__ == "__main__":
    main()
