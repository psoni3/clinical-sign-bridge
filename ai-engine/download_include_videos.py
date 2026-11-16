
import time
import requests
import pandas as pd

from pathlib import Path
from remotezip import RemoteZip

ZENODO_API = "https://zenodo.org/api/records/4010759"

MAPPING_FILE = Path("data/manifests/archive_mapping.csv")
OUTPUT_DIR = Path("data/raw/INCLUDE/videos")
LOG_DIR = Path("data/manifests/download_logs")

MAX_RETRIES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Load required video-to-archive mapping
mapping = pd.read_csv(MAPPING_FILE)

mapping = mapping.drop_duplicates(
    subset=["video_path"]
)

print("Required videos:", len(mapping))
print("Required archives:", mapping["archive"].nunique())

# Retrieve official Zenodo archive URLs
print("\nConnecting to Zenodo...")

response = requests.get(ZENODO_API, timeout=60)
response.raise_for_status()

record = response.json()

archive_urls = {
    item["key"]: item["links"]["self"]
    for item in record["files"]
    if item["key"].lower().endswith(".zip")
}

# Group videos by archive
groups = mapping.groupby("archive")

completed = 0
skipped = 0
failed = []

total = len(mapping)

for archive_name, archive_data in groups:

    archive_url = archive_urls.get(archive_name)

    if not archive_url:
        print(f"\nArchive URL not found: {archive_name}")

        for video_path in archive_data["video_path"]:
            failed.append({
                "archive": archive_name,
                "video_path": video_path,
                "error": "Archive URL not found"
            })

        continue

    print(f"\nProcessing archive: {archive_name}")

    # Retry the archive if any video extraction fails
    for attempt in range(1, MAX_RETRIES + 1):

        try:
            with RemoteZip(archive_url) as remote_zip:

                for video_path in archive_data["video_path"]:

                    # Protect against unsafe archive paths
                    relative_path = Path(video_path)

                    if (
                        relative_path.is_absolute()
                        or ".." in relative_path.parts
                    ):
                        failed.append({
                            "archive": archive_name,
                            "video_path": video_path,
                            "error": "Unsafe video path"
                        })
                        continue

                    output_path = OUTPUT_DIR / relative_path

                    # Skip an already downloaded non-empty video
                    if output_path.is_file() and output_path.stat().st_size > 0:
                        skipped += 1
                        completed += 1

                        print(
                            f"[{completed}/{total}] "
                            f"Already exists: {video_path}"
                        )
                        continue

                    try:
                        video_data = remote_zip.read(video_path)

                        if not video_data:
                            raise ValueError("Downloaded video is empty")

                        output_path.parent.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        # Write to a temporary file first
                        temp_path = output_path.with_suffix(
                            output_path.suffix + ".tmp"
                        )

                        temp_path.write_bytes(video_data)
                        temp_path.replace(output_path)

                        completed += 1

                        print(
                            f"[{completed}/{total}] "
                            f"Downloaded: {video_path} "
                            f"({len(video_data) / (1024 * 1024):.2f} MB)"
                        )

                    except Exception as error:
                        raise RuntimeError(
                            f"Failed to extract {video_path}: {error}"
                        ) from error

            # All videos in this archive were processed
            break

        except Exception as error:

            print(
                f"Archive attempt {attempt}/{MAX_RETRIES} failed: {error}"
            )

            if attempt == MAX_RETRIES:
                # Record videos still missing from this archive
                for video_path in archive_data["video_path"]:

                    output_path = OUTPUT_DIR / Path(video_path)

                    if not output_path.is_file() or output_path.stat().st_size == 0:
                        failed.append({
                            "archive": archive_name,
                            "video_path": video_path,
                            "error": str(error)
                        })

            else:
                time.sleep(5 * attempt)

    # Save failure report after every archive
    pd.DataFrame(failed).to_csv(
        LOG_DIR / "failed_downloads.csv",
        index=False
    )

    print(
        f"Progress: {completed}/{total} "
        f"| Skipped existing: {skipped} "
        f"| Failures: {len(failed)}"
    )

# Final summary
print("\n" + "=" * 55)
print("DOWNLOAD SUMMARY")
print("=" * 55)

print("Required videos:", total)
print("Downloaded or already present:", completed)
print("Skipped existing videos:", skipped)
print("Failed videos:", len(failed))

all_files = list(OUTPUT_DIR.rglob("*.MOV"))

print("MOV files present:", len(all_files))

total_bytes = sum(
    file.stat().st_size for file in all_files
)

print(f"Total local video size: {total_bytes / (1024 ** 3):.2f} GB")

print("\nFailure report:")
print(LOG_DIR / "failed_downloads.csv")
