
import numpy as np

FILE = (
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817.npz"
)

data = np.load(FILE)

landmarks = data["landmarks"]
detected = data["hand_detected"]

hand_counts = detected.sum(axis=1)

print("\nONE-FRAME GAP ASSOCIATION TEST")
print("=" * 65)

tested = 0

for start in range(len(detected) - 2):

    end = start + 2

    # Both hands detected before and after exactly one missing frame.
    if hand_counts[start] != 2 or hand_counts[end] != 2:
        continue

    previous = landmarks[start]
    current = landmarks[end]

    same = (
        np.linalg.norm(previous[0] - current[0], axis=1).mean()
        +
        np.linalg.norm(previous[1] - current[1], axis=1).mean()
    )

    swapped = (
        np.linalg.norm(previous[0] - current[1], axis=1).mean()
        +
        np.linalg.norm(previous[1] - current[0], axis=1).mean()
    )

    winner = "SAME" if same < swapped else "SWAPPED"

    print(
        f"Frames {start} -> {end}: "
        f"same={same:.4f}, "
        f"swapped={swapped:.4f}, "
        f"preferred={winner}"
    )

    tested += 1

print("\nTotal one-frame-gap cases:", tested)
