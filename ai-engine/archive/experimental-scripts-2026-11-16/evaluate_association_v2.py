
import numpy as np

FILE = (
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated_v2.npz"
)

data = np.load(FILE)

landmarks = data["tracked_landmarks"]
detected = data["tracked_detected"]

MAX_GAP = 5

print("\nTRACK IDENTITY CONTINUITY")
print("=" * 60)

for track in range(2):

    indices = np.where(detected[:, track])[0]

    displacements = []
    gaps = []

    for previous, current in zip(indices[:-1], indices[1:]):

        gap = current - previous

        if gap > MAX_GAP:
            continue

        distance = np.linalg.norm(
            landmarks[previous, track]
            - landmarks[current, track],
            axis=1
        ).mean()

        displacements.append(distance)
        gaps.append(gap)

    print(f"\nTrack {track + 1}")
    print("Detected observations:", len(indices))
    print("Comparable pairs:", len(displacements))

    if displacements:
        print("Mean displacement:", round(float(np.mean(displacements)), 5))
        print("Median displacement:", round(float(np.median(displacements)), 5))
        print("95th percentile:", round(float(np.percentile(displacements, 95)), 5))
        print("Maximum gap:", max(gaps))

print("\nOVERALL")
print("Frames with at least one tracked hand:",
      int(detected.any(axis=1).sum()))

print("Frames with two tracked hands:",
      int((detected.sum(axis=1) == 2).sum()))

print("Ambiguous frames:",
      int(data["ambiguous"].sum()))
