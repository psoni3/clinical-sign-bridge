
from pathlib import Path
import cv2
import numpy as np

VIDEO_ROOT = Path("data/raw/INCLUDE/videos")
FEATURE_ROOT = Path("artifacts/features/INCLUDE")
OUTPUT = Path("artifacts/mediapipe/problem_video_inspection")

OUTPUT.mkdir(parents=True, exist_ok=True)

samples = [
    ("test/People/77. Boy/MVI_5134.MOV", "left_zero"),
    ("train/Colours/55. White/MVI_8620.MP4", "left_zero_white"),
    ("test/Home/40. Paint/MVI_8817.MP4", "right_low"),
]

for relative, name in samples:
    parts = Path(relative).parts
    split, *video_parts = parts

    video_path = VIDEO_ROOT.joinpath(*video_parts)
    feature_path = FEATURE_ROOT / split / Path(*video_parts)
    feature_path = Path(str(feature_path) + ".npz")

    if not video_path.exists() or not feature_path.exists():
        print("Missing:", video_path, feature_path)
        continue

    with np.load(feature_path) as data:
        left = data["left_hand_detected"]
        right = data["right_hand_detected"]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    indices = np.linspace(0, total - 1, 6, dtype=int)

    thumbnails = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.resize(frame, (480, 270))

        label = (
            f"Frame {index} | "
            f"Left: {int(left[index])} | "
            f"Right: {int(right[index])}"
        )

        cv2.putText(
            frame, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

        thumbnails.append(frame)

    cap.release()

    if not thumbnails:
        continue

    sheet = np.vstack([
        np.hstack(thumbnails[:3]),
        np.hstack(thumbnails[3:])
    ])

    output_file = OUTPUT / f"{name}.jpg"
    cv2.imwrite(str(output_file), sheet)

    print("Generated:", output_file)

print("\nInspection images generated.")
