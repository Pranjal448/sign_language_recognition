"""Step 8: train the LSTM.

Standard supervised training loop: cross-entropy loss, Adam optimizer,
track train vs. validation loss/accuracy every epoch so overfitting is
visible as it happens (val loss rising while train loss keeps falling).

Run:
  ./venv/bin/python train.py
Produces:
  model.pt             - weights from the epoch with the best val accuracy
  training_curves.png  - loss/accuracy curves for the writeup
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_split
from model import GestureLSTM

SEED = 42
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += xb.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"train: {X_train.shape}, val: {X_val.shape}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = GestureLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pt")
            marker = "  <- saved (best val acc so far)"
        else:
            marker = ""

        print(f"epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.3f} (model.pt saved at that epoch)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png")


if __name__ == "__main__":
    main()
