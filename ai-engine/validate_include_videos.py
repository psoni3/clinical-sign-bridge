
from pathlib import Path
from collections import Counter
import pandas as pd
import cv2

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = BASE_DIR / "data" / "manifests" / "cleaned"
VIDEO_DIR = BASE_DIR / "data" / "raw" / "INCLUDE" / "videos"
REPORT_DIR = BASE_DIR / "data" / "manifests" / "validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load all cleaned manifests
manifest_files = ["train.csv", "validation.csv", "test.csv"]
frames = []

for filename in manifest_files:
    path = MANIFEST_DIR / filename
    df = pd.read_csv(path)
    df["source_split"] = filename.replace(".csv", "")
    frames.append(df)

manifest = pd.concat(frames, ignore_index=True)

if "video_path" not in manifest.columns:
    raise ValueError(f"video_path column missing. Found: {list(manifest.columns)}")

manifest["video_path"] = (
    manifest["video_path"].astype(str).str.replace("\\", "/", regex=False)
)

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)
print(f"Manifest records: {len(manifest)}")
print(f"Unique video paths: {manifest['video_path'].nunique()}")

duplicates = manifest[manifest.duplicated("video_path", keep=False)]
print(f"Duplicate manifest records: {len(duplicates)}")

results = []

for index, row in manifest.iterrows():
    relative_path = Path(row["video_path"])
    video_path = VIDEO_DIR / relative_path

    result = {
        "video_path": row["video_path"],
        "label": row.get("label", ""),
        "split": row["source_split"],
        "exists": video_path.is_file(),
        "size_mb": 0,
        "readable": False,
        "frames": 0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "duration_seconds": 0,
        "error": ""
    }

    if not video_path.is_file():
        result["error"] = "FILE_MISSING"
        results.append(result)
        continue

    result["size_mb"] = round(video_path.stat().st_size / (1024 * 1024), 2)

    if result["size_mb"] == 0:
        result["error"] = "EMPTY_FILE"
        results.append(result)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        result["error"] = "CANNOT_OPEN"
        cap.release()
        results.append(result)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    decode_error = False

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame is None:
            decode_error = True
            break

        frame_count += 1

    cap.release()

    result["frames"] = frame_count
    result["fps"] = round(fps, 2) if fps else 0
    result["width"] = width
    result["height"] = height
    result["duration_seconds"] = (
        round(frame_count / fps, 2) if fps and fps > 0 else 0
    )

    if frame_count == 0:
        result["error"] = "NO_DECODABLE_FRAMES"
    elif decode_error:
        result["error"] = "FRAME_DECODE_ERROR"
    elif fps <= 0 or width <= 0 or height <= 0:
        result["error"] = "INVALID_VIDEO_METADATA"
    else:
        result["readable"] = True

    results.append(result)

    if (index + 1) % 50 == 0:
        print(f"Validated {index + 1}/{len(manifest)}")

report = pd.DataFrame(results)
report_path = REPORT_DIR / "video_validation_report.csv"
report.to_csv(report_path, index=False)

# Compare local files with the manifest
expected_paths = set(manifest["video_path"])
local_files = [
    p for p in VIDEO_DIR.rglob("*")
    if p.is_file()
]

local_relative_paths = {
    p.relative_to(VIDEO_DIR).as_posix()
    for p in local_files
}

extra_files = local_relative_paths - expected_paths
missing_files = expected_paths - local_relative_paths

extensions = Counter(
    p.suffix.lower() for p in local_files
)

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Required manifest records: {len(manifest)}")
print(f"Unique required videos: {len(expected_paths)}")
print(f"Local video files: {len(local_files)}")
print(f"Missing files: {len(missing_files)}")
print(f"Extra local files: {len(extra_files)}")
print(f"Readable videos: {report['readable'].sum()}")
print(f"Unreadable videos: {(~report['readable']).sum()}")
print(f"Total local size: {sum(p.stat().st_size for p in local_files) / (1024**3):.2f} GB")

print("\nFile extensions:")
for extension, count in sorted(extensions.items()):
    print(f"  {extension or '[no extension]'}: {count}")

print("\nError breakdown:")
print(report.loc[~report["readable"], "error"].value_counts().to_string())

if missing_files:
    pd.DataFrame({"video_path": sorted(missing_files)}).to_csv(
        REPORT_DIR / "missing_videos.csv", index=False
    )

if extra_files:
    pd.DataFrame({"video_path": sorted(extra_files)}).to_csv(
        REPORT_DIR / "extra_videos.csv", index=False
    )

print(f"\nDetailed report: {report_path}")
print("Validation complete.")
