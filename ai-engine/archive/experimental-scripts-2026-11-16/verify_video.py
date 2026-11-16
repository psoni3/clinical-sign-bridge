
import cv2
from pathlib import Path

video_dir = Path("data/raw/INCLUDE/videos")

videos = list(video_dir.rglob("*.MOV"))

if not videos:
    raise FileNotFoundError("No MOV files found")

video_path = videos[0]

print("Testing:", video_path)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

success, frame = cap.read()

cap.release()

print("\nVIDEO DETAILS")
print("Frames:", frame_count)
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("First frame readable:", success)

if not success or frame_count <= 0:
    raise RuntimeError("Video validation failed")

print("\nVIDEO VALIDATION SUCCESSFUL")
