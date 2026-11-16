
import requests
import pandas as pd

ZENODO_API = "https://zenodo.org/api/records/4010759"

mapping = pd.read_csv(
    "data/manifests/archive_mapping.csv"
)

required_archives = set(mapping["archive"].unique())

response = requests.get(ZENODO_API, timeout=60)
response.raise_for_status()

record = response.json()

print("REQUIRED ARCHIVES")
print("-" * 65)

total_bytes = 0

for file_info in record["files"]:
    filename = file_info["key"]

    if filename in required_archives:
        size = file_info["size"]
        total_bytes += size

        print(
            f"{filename:40} "
            f"{size / (1024 ** 3):.3f} GB"
        )

print("-" * 65)
print("Number of archives:", len(required_archives))
print(f"Total download size: {total_bytes / (1024 ** 3):.2f} GB")
