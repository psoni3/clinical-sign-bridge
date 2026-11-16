
import numpy as np

FILE = r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

landmarks = data["landmarks"]
detected = data["hand_detected"]

print("\nHAND SLOT SWAP ANALYSIS")
print("=" * 65)

suspected_swaps = 0
valid_pairs = 0

for frame in range(1, len(detected)):

    # Compare only consecutive frames with two detected hands.
    if not (detected[frame - 1].all() and detected[frame].all()):
        continue

    previous = landmarks[frame - 1]
    current = landmarks[frame]

    # Mean Euclidean landmark displacement.
    same_slot = (
        np.linalg.norm(previous[0] - current[0], axis=1).mean()
        +
        np.linalg.norm(previous[1] - current[1], axis=1).mean()
    )

    swapped_slot = (
        np.linalg.norm(previous[0] - current[1], axis=1).mean()
        +
        np.linalg.norm(previous[1] - current[0], axis=1).mean()
    )

    valid_pairs += 1

    if swapped_slot < same_slot * 0.70:
        suspected_swaps += 1

        print(
            f"Possible swap between frames {frame - 1} and {frame}: "
            f"same={same_slot:.4f}, swapped={swapped_slot:.4f}"
        )

print("\nSUMMARY")
print("Valid consecutive two-hand pairs:", valid_pairs)
print("Possible slot swaps:", suspected_swaps)
