
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

VIDEO = Path(
    r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"
)

OUTPUT = Path(
    r"artifacts/mediapipe/standalone_experiment/MVI_8817.npz"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands

detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(str(VIDEO))

all_landmarks = []
all_detected = []
all_handedness = []
all_confidence = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    # Two hand slots, 21 landmarks, XYZ coordinates
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    detected = np.zeros(2, dtype=bool)
    handedness = np.full(2, "", dtype="<U10")
    confidence = np.zeros(2, dtype=np.float32)

    if result.multi_hand_landmarks:
        for i, hand in enumerate(result.multi_hand_landmarks):
            if i >= 2:
                break

            points = np.array(
                [[p.x, p.y, p.z] for p in hand.landmark],
                dtype=np.float32
            )

            landmarks[i] = points
            detected[i] = True

            classification = result.multi_handedness[i].classification[0]

            handedness[i] = classification.label
            confidence[i] = classification.score

    all_landmarks.append(landmarks)
    all_detected.append(detected)
    all_handedness.append(handedness)
    all_confidence.append(confidence)

cap.release()
detector.close()

np.savez_compressed(
    OUTPUT,
    landmarks=np.array(all_landmarks),
    hand_detected=np.array(all_detected),
    handedness=np.array(all_handedness),
    confidence=np.array(all_confidence)
)

print("\nEXTRACTION COMPLETED")
print("Output:", OUTPUT)
print("Frames:", len(all_landmarks))
print("Landmarks shape:", np.array(all_landmarks).shape)
print("Detected hands:", np.array(all_detected).sum())
print("Left labels:", (np.array(all_handedness) == "Left").sum())
print("Right labels:", (np.array(all_handedness) == "Right").sum())
