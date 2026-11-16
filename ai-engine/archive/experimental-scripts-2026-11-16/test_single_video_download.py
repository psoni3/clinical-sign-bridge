
import requests
import pandas as pd

from pathlib import Path
from remotezip import RemoteZip

ZENODO_API = "https://zenodo.org/api/records/4010759"

# Load the archive-to-video mapping
mapping = pd.read_csv(
    "data/manifests/archive_mapping.csv"
)

# Select the first video for testing
sample = mapping.iloc[0]

archive_name = sample["archive"]
video_path = sample["video_path"]

print("Selected archive:", archive_name)
print("Selected video:", video_path)

# Find the official archive download URL
response = requests.get(ZENODO_API, timeout=60)
response.raise_for_status()

record = response.json()

archive_info = next(
    item for item in record["files"]
    if item["key"] == archive_name
)

archive_url = archive_info["links"]["self"]

# Download only the selected video using HTTP range requests
with RemoteZip(archive_url) as remote_zip:
    video_data = remote_zip.read(video_path)

# Save the video using its original relative path
output_path = (
    Path("data/raw/INCLUDE/videos") / video_path
)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(video_data)

print("\nDOWNLOAD SUCCESSFUL")
print("Saved to:", output_path)
print("File size:", len(video_data), "bytes")
