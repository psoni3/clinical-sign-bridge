
from pathlib import Path
import pandas as pd

input_dir = Path("data/manifests")
output_dir = Path("data/manifests/cleaned")
output_dir.mkdir(parents=True, exist_ok=True)

files = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

# Load the original filtered manifests
data = pd.concat(
    [pd.read_csv(input_dir / filename) for filename in files.values()],
    ignore_index=True
)

print("Original records:", len(data))

# Find paths that have more than one distinct label
label_counts = data.groupby("video_path")["label"].nunique()
conflicting_paths = label_counts[label_counts > 1].index

conflicts = data[data["video_path"].isin(conflicting_paths)].copy()

# Save conflicting records separately for future investigation
conflicts.to_csv(output_dir / "quarantined_conflicts.csv", index=False)

# Exclude every record associated with a conflicting path
cleaned = data[~data["video_path"].isin(conflicting_paths)].copy()

# Remove repeated records for the same video path
cleaned = cleaned.drop_duplicates(
    subset=["video_path"],
    keep="first"
)

# Save one manifest per split
for split, filename in files.items():
    split_data = cleaned[cleaned["split"] == split].copy()
    split_data.to_csv(output_dir / filename, index=False)

    print(f"{split}: {len(split_data)} records")

print("\nCLEANING SUMMARY")
print("Original records:", len(data))
print("Conflicting paths:", len(conflicting_paths))
print("Quarantined records:", len(conflicts))
print("Cleaned unique videos:", len(cleaned))
print("Unique labels:", cleaned["label"].nunique())
print("Duplicate paths remaining:", cleaned["video_path"].duplicated().sum())

print("\nCleaned manifests saved to:", output_dir)
