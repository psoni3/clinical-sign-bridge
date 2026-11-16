
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MANIFEST = BASE_DIR / "data/manifests/frozen/train.csv"
VIDEO_DIR = BASE_DIR / "data/raw/INCLUDE/videos"
OUTPUT_DIR = BASE_DIR / "artifacts/mediapipe"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select the first training video
df = pd.read_csv(MANIFEST)
video_relative_path = str(df.iloc[0]["video_path"])
video_path = VIDEO_DIR / video_relative_path

print("Video:", video_relative_path)

if not video_path.exists():
    raise FileNotFoundError(video_path)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)

pose_count = 0
left_hand_count = 0
right_hand_count = 0
processed_frames = 0

preview_frame = None

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while processed_frames < 30:
        success, frame = cap.read()

        if not success:
            break

        # Resize to reduce CPU processing time
        height, width = frame.shape[:2]
        scale = 640 / width
        frame = cv2.resize(
            frame,
            (640, int(height * scale))
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)

        if results.pose_landmarks:
            pose_count += 1

        if results.left_hand_landmarks:
            left_hand_count += 1

        if results.right_hand_landmarks:
            right_hand_count += 1

        # Keep an annotated preview
        if processed_frames == 14:
            preview_frame = frame.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    preview_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    preview_frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    preview_frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

        processed_frames += 1

cap.release()

if preview_frame is not None:
    preview_path = OUTPUT_DIR / "mediapipe_preview.jpg"
    cv2.imwrite(str(preview_path), preview_frame)
    print("Preview saved:", preview_path)

print("\n" + "=" * 50)
print("MEDIAPIPE SMOKE TEST")
print("=" * 50)
print("Frames processed:", processed_frames)
print("Pose detected:", pose_count)
print("Left hand detected:", left_hand_count)
print("Right hand detected:", right_hand_count)

if processed_frames == 0:
    print("RESULT: FAILED - No frames processed")
elif pose_count == 0:
    print("RESULT: WARNING - No pose landmarks detected")
else:
    print("RESULT: BASIC DETECTION SUCCESSFUL")
