
import cv2
import mediapipe as mp

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"

mp_holistic = mp.solutions.holistic


def test_resolution(target_width):
    cap = cv2.VideoCapture(VIDEO)

    detected_left = 0
    detected_right = 0
    total = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            success, frame = cap.read()

            if not success:
                break

            if target_width is not None:
                height, width = frame.shape[:2]
                new_height = int(height * target_width / width)
                frame = cv2.resize(
                    frame, (target_width, new_height)
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            total += 1
            detected_left += (
                results.left_hand_landmarks is not None
            )
            detected_right += (
                results.right_hand_landmarks is not None
            )

    cap.release()

    print(f"\nResolution: {target_width or 'Original'}")
    print("Frames:", total)
    print(f"Left hand: {detected_left / total * 100:.2f}%")
    print(f"Right hand: {detected_right / total * 100:.2f}%")


test_resolution(640)
test_resolution(None)
