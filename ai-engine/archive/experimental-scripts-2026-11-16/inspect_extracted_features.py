
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = BASE_DIR / "artifacts/features/smoke_test"

files = list(FEATURE_DIR.glob("*.npz"))

if not files:
    raise FileNotFoundError("No NPZ feature file found")

feature_path = files[0]

with np.load(feature_path, allow_pickle=False) as data:
    landmarks = data["landmarks"]
    pose = data["pose_detected"]
    left = data["left_hand_detected"]
    right = data["right_hand_detected"]

    print("=" * 55)
    print("EXTRACTED FEATURE VALIDATION")
    print("=" * 55)

    print("File:", feature_path.name)
    print("Landmarks shape:", landmarks.shape)
    print("Pose mask shape:", pose.shape)
    print("Left hand mask shape:", left.shape)
    print("Right hand mask shape:", right.shape)

    print("\nData type:", landmarks.dtype)
    print("NaN values:", np.isnan(landmarks).sum())
    print("Infinite values:", np.isinf(landmarks).sum())

    print("\nPose detection:", pose.mean() * 100, "%")
    print("Left hand detection:", left.mean() * 100, "%")
    print("Right hand detection:", right.mean() * 100, "%")

    print("\nCoordinate range:")
    print("Minimum:", landmarks.min())
    print("Maximum:", landmarks.max())

    assert landmarks.ndim == 3
    assert landmarks.shape[1:] == (75, 3)
    assert len(pose) == len(landmarks)
    assert len(left) == len(landmarks)
    assert len(right) == len(landmarks)
    assert not np.isnan(landmarks).any()
    assert not np.isinf(landmarks).any()

    print("\nRESULT: FEATURE FILE VALIDATION PASSED")
