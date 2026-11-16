
from pathlib import Path
import numpy as np

ROOT = Path("artifacts/features/INCLUDE")

files = list(ROOT.rglob("*.npz"))

total_frames = 0
pose_detected = 0
left_detected = 0
right_detected = 0
failed = []

for file in files:
    try:
        with np.load(file) as data:
            pose = data["pose_detected"]
            left = data["left_hand_detected"]
            right = data["right_hand_detected"]

            total_frames += len(pose)
            pose_detected += int(pose.sum())
            left_detected += int(left.sum())
            right_detected += int(right.sum())

    except Exception as exc:
        failed.append((str(file), str(exc)))

print("\nLANDMARK QUALITY REPORT")
print("=" * 50)
print("Feature files:", len(files))
print("Total frames:", total_frames)

if total_frames:
    print(f"Pose detection: {pose_detected / total_frames * 100:.2f}%")
    print(f"Left hand detection: {left_detected / total_frames * 100:.2f}%")
    print(f"Right hand detection: {right_detected / total_frames * 100:.2f}%")

print("Failed files:", len(failed))

for file, error in failed[:10]:
    print("FAILED:", file, error)
