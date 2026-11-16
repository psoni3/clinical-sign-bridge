
from pathlib import Path
import pandas as pd

manifest_dir = Path("data/manifests/cleaned")

files = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

data = pd.concat(
    [pd.read_csv(manifest_dir / f) for f in files],
    ignore_index=True
)

print("REQUIRED VIDEO CATEGORIES:")
categories = sorted(data["parent_label"].unique())

for category in categories:
    count = len(data[data["parent_label"] == category])
    print(f"{category}: {count} videos")

print("\nTOTAL CATEGORIES:", len(categories))
print("TOTAL VIDEO RECORDS:", len(data))
