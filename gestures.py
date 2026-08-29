"""Step 2: the gesture label set for this project.

Kept as a single source of truth — imported by the data collection,
training, and inference scripts so the label list never drifts out of sync.

Switched from a custom command set to 10 words from the public LSA64
(Argentinian Sign Language) dataset — see process_lsa64.py for the sign-id
mapping and preprocessing.
"""

GESTURES = [
    "red",
    "green",
    "milk",
    "water",
    "food",
    "name",
    "thanks",
    "help",
    "buy",
    "run",
]
