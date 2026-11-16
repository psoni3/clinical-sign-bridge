
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

original_detected = original["hand_detected"]
tracked_detected = result["tracked_detected"]
track_ids = result["track_ids"]
ambiguous = result["ambiguous"]
costs = result["association_cost"]

print("\nASSOCIATION V2 VALIDATION")
print("=" * 75)

print("Frame | Original | Tracked | Track IDs | Ambiguous | Costs")
print("-" * 75)

for frame in range(len(original_detected)):

    original_count = int(original_detected[frame].sum())
    tracked_count = int(tracked_detected[frame].sum())

    if original_count > 0:
        print(
            f"{frame:5d} | "
            f"{original_count:8d} | "
            f"{tracked_count:7d} | "
            f"{track_ids[frame].tolist()} | "
            f"{str(bool(ambiguous[frame])):9s} | "
            f"{np.round(costs[frame], 4).tolist()}"
        )

print("\nKEY FRAME INSPECTION")
for frame in [14, 15, 16, 17, 22, 24, 38, 39, 40, 41, 54, 56]:
    if frame < len(original_detected):
        print(
            f"Frame {frame}: "
            f"original={original_detected[frame].sum()}, "
            f"tracked={tracked_detected[frame].sum()}, "
            f"IDs={track_ids[frame].tolist()}, "
            f"ambiguous={ambiguous[frame]}"
        )
