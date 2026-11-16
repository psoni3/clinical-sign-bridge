
import json
import urllib.request

url = "https://zenodo.org/api/records/4010759"

print("Connecting to Zenodo...")

request = urllib.request.Request(
    url,
    headers={"User-Agent": "ClinicalSignBridge-Dataset-Research/1.0"}
)

with urllib.request.urlopen(request, timeout=60) as response:
    record = json.loads(response.read().decode("utf-8"))

files = record.get("files", [])

if not files:
    print("No files found in the Zenodo record.")
    raise SystemExit(1)

print("\nOFFICIAL INCLUDE DATASET ARCHIVES")
print("-" * 75)

total_bytes = 0

for file in files:
    name = file.get("key", "Unknown")
    size = file.get("size", 0)

    total_bytes += size

    print(f"{name:45} {size / (1024 ** 3):8.3f} GB")

print("-" * 75)
print("Number of files:", len(files))
print(f"Total size: {total_bytes / (1024 ** 3):.2f} GB")

print("\nInspection completed. No files were downloaded.")
