
import cv2
import mediapipe as mp
from pathlib import Path

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"

OUTPUT = Path("artifacts/mediapipe/handedness_inspection")
OUTPUT.mkdir(parents=True, exist_ok=True)

FRAMES = [
    3, 4, 5,
    17, 18, 19,
    22, 23, 24,
    52, 53, 54,
    58, 59, 60,
    63, 64, 65, 66
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO)
frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number in FRAMES:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.multi_hand_landmarks:
            for i, hand in enumerate(result.multi_hand_landmarks):
                classification = result.multi_handedness[i].classification[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand.landmark[0]
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])

                label = (
                    f"{classification.label} "
                    f"{classification.score:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        output_file = OUTPUT / f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(str(output_file), frame)

        print("Saved:", output_file)

    frame_number += 1

cap.release()
detector.close()

print("\nInspection images generated.")
