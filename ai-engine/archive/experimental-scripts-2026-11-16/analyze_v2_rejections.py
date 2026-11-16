
import numpy as np

SOURCE = (
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817.npz"
)

OUTPUT = (
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated_v2.npz"
)

original = np.load(SOURCE)
result = np.load(OUTPUT)

detected = original["hand_detected"]
tracked = result["tracked_detected"]
ambiguous = result["ambiguous"]

counts = detected.sum(axis=1)
tracked_counts = tracked.sum(axis=1)

print("\nV2 REJECTION ANALYSIS")
print("=" * 60)

for count, label in [(0, "Zero-hand"), (1, "One-hand"), (2, "Two-hand")]:

    frames = np.where(counts == count)[0]
    rejected = frames[tracked_counts[frames] == 0]

    print(f"\n{label} frames: {len(frames)}")
    print("Rejected:", len(rejected))
    print("Rejected frame numbers:", rejected.tolist())

print("\nREJECTED FRAMES WITH AMBIGUITY FLAG")
print(np.where(ambiguous & (tracked_counts == 0))[0].tolist())

print("\nUNTRACKED FRAMES WITHOUT AMBIGUITY FLAG")
print(np.where((tracked_counts == 0) & ~ambiguous)[0].tolist())
