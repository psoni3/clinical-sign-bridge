
import numpy as np

FILE = (
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817.npz"
)

data = np.load(FILE)

detected = data["hand_detected"]
associated_data = np.load(
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated.npz"
)

associated = associated_data["detected"]
ambiguous = associated_data["ambiguous"]

hand_counts = detected.sum(axis=1)

print("\nASSOCIATION DIAGNOSTICS")
print("=" * 55)

print("Total frames:", len(detected))
print("Zero-hand frames:", np.sum(hand_counts == 0))
print("One-hand frames:", np.sum(hand_counts == 1))
print("Two-hand frames:", np.sum(hand_counts == 2))

print("\nAssociation output")
print("Frames with at least one associated hand:",
      np.sum(associated.any(axis=1)))
print("Frames with two associated hands:",
      np.sum(associated.sum(axis=1) == 2))
print("Ambiguous frames:", np.sum(ambiguous))

# Consecutive two-hand frame pairs
pairs = 0

for i in range(1, len(detected)):
    if hand_counts[i - 1] == 2 and hand_counts[i] == 2:
        pairs += 1

print("\nConsecutive two-hand frame pairs:", pairs)

# Distribution of gaps between two-hand frames
two_hand_indices = np.where(hand_counts == 2)[0]

if len(two_hand_indices) > 1:
    gaps = np.diff(two_hand_indices)

    print("Median gap between two-hand frames:",
          np.median(gaps))
    print("Maximum gap between two-hand frames:",
          np.max(gaps))
