
import numpy as np
from pathlib import Path

SOURCE = Path(
    "artifacts/mediapipe/standalone_experiment/MVI_8817.npz"
)

OUTPUT = Path(
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated.npz"
)

data = np.load(SOURCE)

landmarks = data["landmarks"]
detected = data["hand_detected"]

T = len(detected)

associated = np.zeros_like(landmarks)
associated_detected = np.zeros_like(detected)

# 0 = not assigned, 1 = first tracked hand, 2 = second tracked hand
track_ids = np.zeros((T, 2), dtype=np.int8)
ambiguous = np.zeros(T, dtype=bool)

previous_hands = None
previous_ids = None

for frame in range(T):

    current_indices = np.where(detected[frame])[0]

    if len(current_indices) == 0:
        continue

    current_hands = landmarks[frame, current_indices]

    if previous_hands is None:
        for i, slot in enumerate(current_indices):
            associated[frame, i] = landmarks[frame, slot]
            associated_detected[frame, i] = True
            track_ids[frame, i] = i + 1

        previous_hands = current_hands.copy()
        previous_ids = track_ids[frame].copy()
        continue

    if len(previous_hands) == 2 and len(current_hands) == 2:
        same = (
            np.linalg.norm(previous_hands[0] - current_hands[0], axis=1).mean()
            +
            np.linalg.norm(previous_hands[1] - current_hands[1], axis=1).mean()
        )

        swapped = (
            np.linalg.norm(previous_hands[0] - current_hands[1], axis=1).mean()
            +
            np.linalg.norm(previous_hands[1] - current_hands[0], axis=1).mean()
        )

        if min(same, swapped) > 0.30:
            ambiguous[frame] = True
            continue

        if abs(same - swapped) < 0.005:
            ambiguous[frame] = True
            continue

        assignment = [0, 1] if same < swapped else [1, 0]

        for track_index, current_index in enumerate(assignment):
            original_slot = current_indices[current_index]

            associated[frame, track_index] = landmarks[
                frame, original_slot
            ]
            associated_detected[frame, track_index] = True
            track_ids[frame, track_index] = previous_ids[track_index]

    else:
        # Do not guess identity when one or both hands are missing.
        ambiguous[frame] = True
        continue

    previous_hands = associated[frame].copy()
    previous_ids = track_ids[frame].copy()

np.savez_compressed(
    OUTPUT,
    landmarks=associated,
    detected=associated_detected,
    track_ids=track_ids,
    ambiguous=ambiguous
)

print("Output:", OUTPUT)
print("Frames:", T)
print("Ambiguous frames:", int(ambiguous.sum()))
print("Associated frames:", int((associated_detected.any(axis=1)).sum()))
print("Source file unchanged.")
