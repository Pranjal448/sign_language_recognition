"""Step 6: split into train/val/test by session (here: by LSA64 subject),
never mixing the same session across splits — otherwise the model could
"cheat" by recognizing a specific person's hand rather than the gesture.

Splitting by subject is actually the harder, more meaningful test here:
the model has to generalize to a person it never saw during training, not
just a different sitting with the same person.

Run:
  ./venv/bin/python split_data.py
Produces data/splits.csv (same rows as metadata.csv, plus a split column).
"""
import csv
import random
from collections import defaultdict

METADATA_PATH = "data/metadata.csv"
SPLITS_PATH = "data/splits.csv"
SEED = 42
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
# remaining goes to test


def main():
    with open(METADATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    sessions = sorted(set(r["session_id"] for r in rows))
    rng = random.Random(SEED)
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = max(1, round(n * TRAIN_FRAC))
    n_val = max(1, round(n * VAL_FRAC))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val -= 1
        n_test = 1

    train_sessions = set(sessions[:n_train])
    val_sessions = set(sessions[n_train:n_train + n_val])
    test_sessions = set(sessions[n_train + n_val:])

    def split_for(session_id):
        if session_id in train_sessions:
            return "train"
        if session_id in val_sessions:
            return "val"
        return "test"

    with open(SPLITS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "gesture", "session_id", "split"])
        for r in rows:
            writer.writerow([r["filepath"], r["gesture"], r["session_id"], split_for(r["session_id"])])

    print(f"{n} sessions total")
    print(f"  train ({len(train_sessions)}): {sorted(train_sessions)}")
    print(f"  val   ({len(val_sessions)}): {sorted(val_sessions)}")
    print(f"  test  ({len(test_sessions)}): {sorted(test_sessions)}")

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[split_for(r["session_id"])][r["gesture"]] += 1

    print()
    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        per_gesture = ", ".join(f"{g}={counts[split].get(g, 0)}" for g in sorted(counts[split]))
        print(f"{split}: {total} sequences  ({per_gesture})")

    print(f"\nSaved {SPLITS_PATH}")


if __name__ == "__main__":
    main()
