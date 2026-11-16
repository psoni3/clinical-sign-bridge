
import numpy as np

FILE = r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

detected = data["hand_detected"]
handedness = data["handedness"]

print("\nHAND SLOT CONSISTENCY")
print("=" * 55)

for slot in range(2):
    labels = handedness[detected[:, slot], slot]

    print(f"\nSlot {slot + 1}")
    print("Detected frames:", len(labels))
    print("Left labels:", int((labels == "Left").sum()))
    print("Right labels:", int((labels == "Right").sum()))

    if len(labels):
        unique, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique, counts):
            print(f"{label}: {count / len(labels) * 100:.2f}%")

# Check whether the same handedness appears in both slots
print("\nFrames with two hands detected:")

two_hand_frames = np.where(detected.sum(axis=1) == 2)[0]

for frame in two_hand_frames:
    print(
        f"Frame {frame}: "
        f"Slot 1={handedness[frame, 0]}, "
        f"Slot 2={handedness[frame, 1]}"
    )

print("\nAnalysis complete.")
