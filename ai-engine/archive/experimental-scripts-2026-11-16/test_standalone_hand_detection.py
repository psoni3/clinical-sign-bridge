
import cv2
import mediapipe as mp

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(VIDEO)

frames = 0
zero_hands = 0
one_hand = 0
two_hands = 0
total_detected_hands = 0

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = (
            len(results.multi_hand_landmarks)
            if results.multi_hand_landmarks else 0
        )

        frames += 1
        total_detected_hands += detected

        if detected == 0:
            zero_hands += 1
        elif detected == 1:
            one_hand += 1
        elif detected == 2:
            two_hands += 1

cap.release()

print("\nSTANDALONE MEDIAPIPE HANDS RESULTS")
print("=" * 45)
print("Frames:", frames)
print("Frames with zero hands:", zero_hands)
print("Frames with one hand:", one_hand)
print("Frames with two hands:", two_hands)
print(
    "Frames with at least one hand:",
    round((one_hand + two_hands) / frames * 100, 2), "%"
)
print(
    "Frames with two hands:",
    round(two_hands / frames * 100, 2), "%"
)
print("Total detected hand instances:", total_detected_hands)
