
from pathlib import Path
import pandas as pd

metadata_dir = Path("data/raw/INCLUDE/metadata")

files = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}

frames = []

for split, filename in files.items():
    df = pd.read_parquet(metadata_dir / filename)
    df["split"] = split

    # Keep only INCLUDE-50 records
    df = df[df["include_50"] == True].copy()
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("\n1. RECORDS PER SPLIT")
print(data.groupby("split").size())

print("\n2. UNIQUE LABELS")
print("Unique labels:", data["label"].nunique())

print("\n3. VIDEOS PER LABEL")
print(data.groupby("label").size().sort_values())

print("\n4. DUPLICATE VIDEO PATHS")
duplicates = data[data.duplicated("video_path", keep=False)]

if duplicates.empty:
    print("No duplicate video paths found.")
else:
    print(duplicates[["video_path", "split", "label"]])

print("\n5. VIDEO PATH OVERLAP BETWEEN SPLITS")
for split_a in files:
    for split_b in files:
        if split_a < split_b:
            paths_a = set(data[data["split"] == split_a]["video_path"])
            paths_b = set(data[data["split"] == split_b]["video_path"])
            overlap = paths_a.intersection(paths_b)

            print(f"{split_a} vs {split_b}: {len(overlap)}")

print("\n6. MISSING VALUES")
print(data.isnull().sum())

print("\n7. SAMPLE VIDEO PATHS")
print(data["video_path"].head(10).to_string(index=False))

# Save the filtered metadata for later use
output_dir = Path("data/manifests")
output_dir.mkdir(parents=True, exist_ok=True)

for split in files:
    split_data = data[data["split"] == split]
    split_data.to_csv(output_dir / f"{split}.csv", index=False)

print("\nFiltered manifests saved in data/manifests/")
