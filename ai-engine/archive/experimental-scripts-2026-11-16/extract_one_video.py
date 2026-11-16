
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MANIFEST = BASE_DIR / "data/manifests/frozen/train.csv"
VIDEO_DIR = BASE_DIR / "data/raw/INCLUDE/videos"
OUTPUT_DIR = BASE_DIR / "artifacts/features/smoke_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

row = df.iloc[0]
video_relative_path = str(row["video_path"])
video_path = VIDEO_DIR / video_relative_path
label = str(row["label"])

if not video_path.is_file():
    raise FileNotFoundError(video_path)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

mp_holistic = mp.solutions.holistic

all_frames = []
pose_masks = []
left_hand_masks = []
right_hand_masks = []

def extract_landmarks(landmarks, expected_count):
    """Return XYZ coordinates or zero placeholders."""
    if landmarks is None:
        return np.zeros((expected_count, 3), dtype=np.float32)

    return np.asarray(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32
    )

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        success, frame = cap.read()

        if not success:
            break

        height, width = frame.shape[:2]
        scale = 640 / width

        frame = cv2.resize(
            frame,
            (640, int(height * scale))
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = holistic.process(rgb)

        pose = extract_landmarks(
            results.pose_landmarks, 33
        )

        left_hand = extract_landmarks(
            results.left_hand_landmarks, 21
        )

        right_hand = extract_landmarks(
            results.right_hand_landmarks, 21
        )

        # 33 + 21 + 21 = 75 landmarks
        # 75 * 3 = 225 coordinate values
        frame_features = np.concatenate(
            [pose, left_hand, right_hand],
            axis=0
        )

        all_frames.append(frame_features)

        pose_masks.append(
            int(results.pose_landmarks is not None)
        )

        left_hand_masks.append(
            int(results.left_hand_landmarks is not None)
        )

        right_hand_masks.append(
            int(results.right_hand_landmarks is not None)
        )

cap.release()

if not all_frames:
    raise RuntimeError("No frames were processed")

landmarks_array = np.stack(all_frames).astype(np.float32)

output_path = (
    OUTPUT_DIR /
    f"{Path(video_relative_path).stem}.npz"
)

np.savez_compressed(
    output_path,
    landmarks=landmarks_array,
    pose_detected=np.asarray(pose_masks, dtype=np.uint8),
    left_hand_detected=np.asarray(left_hand_masks, dtype=np.uint8),
    right_hand_detected=np.asarray(right_hand_masks, dtype=np.uint8),
    fps=np.float32(fps),
    label=np.asarray(label),
    source_video=np.asarray(video_relative_path)
)

print("=" * 55)
print("LANDMARK EXTRACTION TEST")
print("=" * 55)
print("Source video:", video_relative_path)
print("Label:", label)
print("Frames extracted:", len(all_frames))
print("Feature array shape:", landmarks_array.shape)
print("Expected shape: (frames, 75, 3)")
print("Flattened feature count per frame:", landmarks_array.shape[1] * 3)
print("Pose detection rate:", round(np.mean(pose_masks) * 100, 2), "%")
print("Left hand detection rate:", round(np.mean(left_hand_masks) * 100, 2), "%")
print("Right hand detection rate:", round(np.mean(right_hand_masks) * 100, 2), "%")
print("Output:", output_path)
