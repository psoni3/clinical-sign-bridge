
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
    path = metadata_dir / filename
    df = pd.read_parquet(path)
    df["split"] = split
    frames.append(df)

    print(f"\n{split.upper()}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

data = pd.concat(frames, ignore_index=True)

print("\nTOTAL RECORDS:", len(data))

print("\nINCLUDE-50 DISTRIBUTION:")
print(data["include_50"].value_counts(dropna=False))

include50 = data[data["include_50"] == True]

print("\nINCLUDE-50 VIDEO COUNT:", len(include50))
print("\nINCLUDE-50 SIGN LABELS:")
print(sorted(include50["label"].dropna().unique()))

print("\nSAMPLE RECORDS:")
print(include50.head(5).to_string(index=False))
