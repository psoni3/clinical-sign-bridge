
import numpy as np

FILE = r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

landmarks = data["landmarks"]
detected = data["hand_detected"]

print("\nTEMPORAL LANDMARK QUALITY")
print("=" * 55)

for hand_idx, hand_name in enumerate(["Hand slot 1", "Hand slot 2"]):

    valid_frames = np.where(detected[:, hand_idx])[0]

    movements = []

    for i in range(1, len(valid_frames)):
        previous_frame = valid_frames[i - 1]
        current_frame = valid_frames[i]

        # Ignore comparisons across missing-frame gaps
        if current_frame != previous_frame + 1:
            continue

        previous = landmarks[previous_frame, hand_idx]
        current = landmarks[current_frame, hand_idx]

        displacement = np.linalg.norm(current - previous, axis=1)

        # Average movement across 21 landmarks
        movements.append(float(displacement.mean()))

    print(f"\n{hand_name}")
    print("Detected frames:", len(valid_frames))
    print("Consecutive frame pairs:", len(movements))

    if movements:
        movements = np.array(movements)

        print("Mean displacement:", round(float(movements.mean()), 5))
        print("Median displacement:", round(float(np.median(movements)), 5))
        print("95th percentile:", round(float(np.percentile(movements, 95)), 5))
        print("Maximum displacement:", round(float(movements.max()), 5))
        print("Large jumps (>0.15):", int((movements > 0.15).sum()))

print("\nAnalysis complete.")
