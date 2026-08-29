"""Step 7: the LSTM classifier.

Input is a sequence of normalized hand-landmark vectors: (batch, seq_len,
63) — 63 = 21 landmarks x (x, y, z). The LSTM reads the sequence frame by
frame and its final hidden state summarizes the motion; a small linear
layer maps that summary to gesture class logits (CrossEntropyLoss applies
softmax internally during training, so the model itself just outputs raw
logits).

Kept deliberately small: this is a low-dimensional input (63 numbers, not
image pixels) over a short sequence (30 frames), so a single LSTM layer
with a modest hidden size is enough — no need for depth here.
"""
import torch
import torch.nn as nn

from gestures import GESTURES

INPUT_SIZE = 63  # 21 landmarks x (x, y, z)
NUM_CLASSES = len(GESTURES)


class GestureLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=1,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) -- take the last layer's
        # final hidden state as the sequence summary
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)  # (batch, num_classes) raw logits


if __name__ == "__main__":
    # quick shape sanity check with a random batch matching dataset.py's output shape
    model = GestureLSTM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GestureLSTM: {n_params:,} parameters")

    dummy = torch.randn(8, 30, INPUT_SIZE)  # (batch=8, seq_len=30, features=63)
    logits = model(dummy)
    print("Input shape:", dummy.shape)
    print("Output shape:", logits.shape, "(expected (8,", NUM_CLASSES, "))")
    assert logits.shape == (8, NUM_CLASSES)
    print("OK")
