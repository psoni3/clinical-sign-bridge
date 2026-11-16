
import numpy as np

FILE = r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"

data = np.load(FILE)

landmarks = data["landmarks"]
detected = data["hand_detected"]
handedness = data["handedness"]
confidence = data["confidence"]

print("\nSTANDALONE LANDMARK VALIDATION")
print("=" * 50)

print("Landmarks shape:", landmarks.shape)
print("Detection mask shape:", detected.shape)
print("Handedness shape:", handedness.shape)
print("Confidence shape:", confidence.shape)

print("\nDetection statistics")
print("Total frames:", len(detected))
print("No hands:", int((~detected.any(axis=1)).sum()))
print("One hand:", int((detected.sum(axis=1) == 1).sum()))
print("Two hands:", int((detected.sum(axis=1) == 2).sum()))

print("\nData integrity")
print("NaN values:", int(np.isnan(landmarks).sum()))
print("Infinite values:", int(np.isinf(landmarks).sum()))
print("Detected hand instances:", int(detected.sum()))

print("\nConfidence statistics")
print("Minimum:", round(float(confidence[detected].min()), 4))
print("Mean:", round(float(confidence[detected].mean()), 4))
print("Maximum:", round(float(confidence[detected].max()), 4))

print("\nHandedness distribution")
print("Left:", int((handedness == "Left").sum()))
print("Right:", int((handedness == "Right").sum()))

print("\nValidation complete.")
