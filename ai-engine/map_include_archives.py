
import requests
import pandas as pd

from pathlib import Path
from remotezip import RemoteZip

ZENODO_API = "https://zenodo.org/api/records/4010759"

manifest_dir = Path("data/manifests/cleaned")

manifest_files = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

data = pd.concat(
    [pd.read_csv(manifest_dir / f) for f in manifest_files],
    ignore_index=True
)

# The exact video paths we need
required_paths = set(
    data["video_path"].str.replace("\\", "/", regex=False)
)

print("Required videos:", len(required_paths))

# Retrieve official archive information
response = requests.get(ZENODO_API, timeout=60)
response.raise_for_status()

record = response.json()

archives = [
    f for f in record["files"]
    if f["key"].lower().endswith(".zip")
]

print("Archives found:", len(archives))

archive_matches = {}
unmatched_paths = set(required_paths)

for archive in archives:
    filename = archive["key"]

    print(f"\nInspecting: {filename}")

    try:
        url = archive["links"]["self"]

        with RemoteZip(url) as remote_zip:
            members = {
                name.replace("\\", "/").lstrip("./")
                for name in remote_zip.namelist()
            }

        matches = required_paths.intersection(members)

        if matches:
            archive_matches[filename] = sorted(matches)
            unmatched_paths.difference_update(matches)

            print("Matching videos:", len(matches))

    except Exception as error:
        print("Could not inspect archive:", error)

print("\nARCHIVE SUMMARY")
print("Archives containing required videos:", len(archive_matches))
print("Required videos found:", len(required_paths) - len(unmatched_paths))
print("Required videos not found:", len(unmatched_paths))

for filename, matches in archive_matches.items():
    print(f"{filename}: {len(matches)} videos")

# Save archive mapping
output_dir = Path("data/manifests")
output_dir.mkdir(parents=True, exist_ok=True)

rows = []

for filename, matches in archive_matches.items():
    for video_path in matches:
        rows.append({
            "archive": filename,
            "video_path": video_path
        })

pd.DataFrame(rows).to_csv(
    output_dir / "archive_mapping.csv",
    index=False
)

pd.DataFrame(
    {"video_path": sorted(unmatched_paths)}
).to_csv(
    output_dir / "unmatched_video_paths.csv",
    index=False
)

print("\nMapping saved to data/manifests/archive_mapping.csv")
