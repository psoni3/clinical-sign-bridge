
import cv2
import mediapipe as mp

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"

mp_holistic = mp.solutions.holistic


def test_mode(static_mode):
    cap = cv2.VideoCapture(VIDEO)

    left_count = 0
    right_count = 0
    total = 0

    with mp_holistic.Holistic(
        static_image_mode=static_mode,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Keep original resolution
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            total += 1
            left_count += (
                results.left_hand_landmarks is not None
            )
            right_count += (
                results.right_hand_landmarks is not None
            )

    cap.release()

    print("\nStatic image mode:", static_mode)
    print("Frames:", total)
    print(f"Left hand: {left_count / total * 100:.2f}%")
    print(f"Right hand: {right_count / total * 100:.2f}%")


test_mode(False)
test_mode(True)
