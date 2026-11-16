
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8820.MP4"

FRAMES_TO_INSPECT = [0, 10, 20, 30, 40, 50, 60]

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO)

frame_number = 0
holistic_detected = []
standalone_detected = []

print("\nSELECTED FRAME DIAGNOSTICS")
print("=" * 60)

OUTPUT_DIR = Path("artifacts/mediapipe/standalone_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h_result = holistic.process(rgb)
    s_result = hands.process(rgb)

    if frame_number in FRAMES_TO_INSPECT:
        annotated = frame.copy()

        if s_result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(
                s_result.multi_hand_landmarks
            ):
                classification = s_result.multi_handedness[i].classification[0]

                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[0]

                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])

                label = (
                    f"{classification.label}: "
                    f"{classification.score:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        output_path = OUTPUT_DIR / f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(str(output_path), annotated)

    h_detected = (
        h_result.left_hand_landmarks is not None
        or h_result.right_hand_landmarks is not None
    )

    s_detected = bool(s_result.multi_hand_landmarks)

    holistic_detected.append(h_detected)
    standalone_detected.append(s_detected)

    if frame_number in FRAMES_TO_INSPECT:
        print(f"\nFrame: {frame_number}")
        print(f"Holistic detected: {h_detected}")
        print(f"Standalone detected: {s_detected}")

        if s_result.multi_hand_landmarks:
            for i, classification in enumerate(
                s_result.multi_handedness
            ):
                hand = classification.classification[0]

                print(
                    f"  Hand {i + 1}: {hand.label}, "
                    f"confidence={hand.score:.3f}"
                )

    frame_number += 1

cap.release()
holistic.close()
hands.close()

h = np.array(holistic_detected)
s = np.array(standalone_detected)

print("\nFINAL COMPARISON")
print("=" * 60)
print("Total frames:", len(h))
print("Both detect:", int((h & s).sum()))
print("Only Holistic:", int((h & ~s).sum()))
print("Only Standalone:", int((~h & s).sum()))
print("Neither:", int((~h & ~s).sum()))
