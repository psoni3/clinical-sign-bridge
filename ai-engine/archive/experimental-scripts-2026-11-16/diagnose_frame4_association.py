
import numpy as np

FILE = "artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

detected = data["hand_detected"]
handedness = data["handedness"]
confidence = data["confidence"]

print("Frame | Detected count | Slot 0 | Slot 1")
print("-" * 55)

for frame in range(0, 7):
    print(
        f"{frame:5} | "
        f"{detected[frame].sum():14} | "
        f"{str(detected[frame, 0]):6} | "
        f"{str(detected[frame, 1]):6}"
    )

    print(
        "      Handedness:", handedness[frame].tolist(),
        "Confidence:", np.round(confidence[frame], 3).tolist()
    )
