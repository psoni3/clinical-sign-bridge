
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")

MANIFEST_DIR = ROOT / "data/manifests/frozen"
FEATURE_DIR = ROOT / "artifacts/features/INCLUDE"

SPLITS = ["train", "validation", "test"]

required_keys = {
    "landmarks",
    "pose_detected",
    "left_hand_detected",
    "right_hand_detected",
    "label",
    "source_video",
}

total_expected = 0
total_valid = 0
errors = []

print("\nFEATURE DATASET VALIDATION")
print("=" * 65)

for split in SPLITS:
    manifest_path = MANIFEST_DIR / f"{split}.csv"
    manifest = pd.read_csv(manifest_path)

    expected = len(manifest)
    valid = 0

    for _, row in manifest.iterrows():
        relative_path = Path(str(row["video_path"]))
        feature_path = (
            FEATURE_DIR / split / (str(relative_path) + ".npz")
        )

        if not feature_path.exists():
            errors.append(f"{split}: Missing {feature_path}")
            continue

        try:
            with np.load(feature_path, allow_pickle=False) as data:
                missing = required_keys - set(data.files)

                if missing:
                    errors.append(
                        f"{split}: {feature_path.name}: "
                        f"Missing keys {sorted(missing)}"
                    )
                    continue

                landmarks = data["landmarks"]
                pose = data["pose_detected"]
                left = data["left_hand_detected"]
                right = data["right_hand_detected"]
                feature_label = str(data["label"].item())

                if landmarks.ndim != 3 or landmarks.shape[1:] != (75, 3):
                    errors.append(
                        f"{split}: Invalid landmark shape "
                        f"{landmarks.shape} in {feature_path.name}"
                    )
                    continue

                frame_count = landmarks.shape[0]

                if not all(
                    len(mask) == frame_count
                    for mask in [pose, left, right]
                ):
                    errors.append(
                        f"{split}: Mask length mismatch "
                        f"in {feature_path.name}"
                    )
                    continue

                if not np.isfinite(landmarks).all():
                    errors.append(
                        f"{split}: NaN/Inf in {feature_path.name}"
                    )
                    continue

                if feature_label != str(row["label"]):
                    errors.append(
                        f"{split}: Label mismatch in {feature_path.name}: "
                        f"{feature_label} != {row['label']}"
                    )
                    continue

                valid += 1

        except Exception as exc:
            errors.append(
                f"{split}: Failed reading {feature_path.name}: {exc}"
            )

    total_expected += expected
    total_valid += valid

    print(
        f"{split:12} Expected: {expected:4} "
        f"Valid: {valid:4} Missing/invalid: {expected - valid:4}"
    )

print("\nSUMMARY")
print("=" * 65)
print("Expected features:", total_expected)
print("Valid features:", total_valid)
print("Errors:", len(errors))

if errors:
    print("\nFIRST 20 ERRORS")
    for error in errors[:20]:
        print("-", error)
else:
    print("\nRESULT: ALL FEATURES VALIDATED")
