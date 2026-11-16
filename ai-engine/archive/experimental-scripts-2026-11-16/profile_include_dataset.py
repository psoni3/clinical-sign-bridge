
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "data" / "manifests" / "validation"

report_path = REPORT_DIR / "video_validation_report.csv"
df = pd.read_csv(report_path)

# Only analyse successfully decoded videos
valid = df[df["readable"] == True].copy()

# 1. Class distribution
class_profile = (
    valid.groupby("label")
    .agg(
        total_videos=("video_path", "count"),
        train=("split", lambda x: (x == "train").sum()),
        validation=("split", lambda x: (x == "validation").sum()),
        test=("split", lambda x: (x == "test").sum()),
    )
    .reset_index()
    .sort_values("label")
)

class_profile.to_csv(
    REPORT_DIR / "class_distribution.csv", index=False
)

# 2. Technical characteristics
technical_profile = valid[
    ["frames", "fps", "width", "height", "duration_seconds", "size_mb"]
].describe().round(2)

technical_profile.to_csv(
    REPORT_DIR / "technical_profile.csv"
)

# 3. Video properties by split
split_profile = (
    valid.groupby("split")
    .agg(
        videos=("video_path", "count"),
        avg_frames=("frames", "mean"),
        min_frames=("frames", "min"),
        max_frames=("frames", "max"),
        avg_fps=("fps", "mean"),
        avg_duration=("duration_seconds", "mean"),
        avg_size_mb=("size_mb", "mean"),
    )
    .round(2)
)

split_profile.to_csv(
    REPORT_DIR / "split_profile.csv"
)

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(class_profile.to_string(index=False))

print("\n" + "=" * 60)
print("TECHNICAL PROFILE")
print("=" * 60)
print(technical_profile.to_string())

print("\n" + "=" * 60)
print("SPLIT PROFILE")
print("=" * 60)
print(split_profile.to_string())

print("\nReports saved to:", REPORT_DIR)
