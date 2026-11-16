
import numpy as np

FILE = r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

detected = data["hand_detected"]
handedness = data["handedness"]
confidence = data["confidence"]

two_hand_frames = np.where(detected.sum(axis=1) == 2)[0]

correct_pattern = 0
same_label = 0
other_pattern = 0

print("\nTWO-HAND HANDEDNESS ANALYSIS")
print("=" * 55)

for frame in two_hand_frames:
    label1 = handedness[frame, 0]
    label2 = handedness[frame, 1]

    if label1 != label2:
        correct_pattern += 1
    elif label1 == label2:
        same_label += 1
        print(
            f"Frame {frame}: SAME LABEL "
            f"({label1}), "
            f"confidence=({confidence[frame, 0]:.3f}, "
            f"{confidence[frame, 1]:.3f})"
        )

print("\nSUMMARY")
print("Frames with two hands:", len(two_hand_frames))
print("Opposite labels:", correct_pattern)
print("Identical labels:", same_label)

if len(two_hand_frames):
    print(
        "Identical-label percentage:",
        round(same_label / len(two_hand_frames) * 100, 2),
        "%"
    )
