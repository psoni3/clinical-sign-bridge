
from pathlib import Path
import pandas as pd

data = pd.concat(
    [
        pd.read_csv("data/manifests/train.csv"),
        pd.read_csv("data/manifests/validation.csv"),
        pd.read_csv("data/manifests/test.csv"),
    ],
    ignore_index=True
)

# Group records by their video path
groups = data.groupby("video_path").agg(
    record_count=("label", "size"),
    unique_labels=("label", "nunique"),
    labels=("label", lambda x: " | ".join(sorted(set(x)))),
    splits=("split", lambda x: " | ".join(sorted(set(x))))
).reset_index()

# Paths occurring more than once
duplicates = groups[groups["record_count"] > 1]

# Paths associated with more than one distinct label
conflicts = groups[groups["unique_labels"] > 1]

# Paths repeated with the same label
same_label_duplicates = groups[
    (groups["record_count"] > 1) &
    (groups["unique_labels"] == 1)
]

print("TOTAL METADATA RECORDS:", len(data))
print("UNIQUE VIDEO PATHS:", data["video_path"].nunique())
print("DUPLICATE PATH GROUPS:", len(duplicates))
print("CONFLICTING LABEL GROUPS:", len(conflicts))
print("SAME-LABEL DUPLICATE GROUPS:", len(same_label_duplicates))

print("\nCONFLICTING VIDEO PATHS:")
print(conflicts.to_string(index=False))

print("\nSAME-LABEL DUPLICATES:")
print(same_label_duplicates.to_string(index=False))

# Save reports for further investigation
output_dir = Path("data/manifests")
conflicts.to_csv(output_dir / "conflicting_video_paths.csv", index=False)
duplicates.to_csv(output_dir / "duplicate_video_paths.csv", index=False)

print("\nReports saved in data/manifests/")
