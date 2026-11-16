
import cv2
import numpy as np
from pathlib import Path

VIDEO = r"data/raw/INCLUDE/videos/Home/40. Paint/MVI_8817.MP4"

SOURCE = Path(
    "artifacts/mediapipe/standalone_experiment/MVI_8817.npz"
)

ASSOCIATED = Path(
    "artifacts/mediapipe/standalone_experiment/"
    "MVI_8817_associated_v2.npz"
)

OUTPUT = Path(
    "artifacts/mediapipe/association_comparison"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

FRAMES = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17]

original = np.load(SOURCE)
tracked = np.load(ASSOCIATED)

original_landmarks = original["landmarks"]
original_detected = original["hand_detected"]

tracked_landmarks = tracked["tracked_landmarks"]
tracked_detected = tracked["tracked_detected"]

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_hands(image, hands, masks, prefix):
    image = image.copy()
    height, width = image.shape[:2]

    colors = [(0, 255, 0), (255, 100, 0)]

    for hand_index in range(2):
        if not masks[hand_index]:
            continue

        points = hands[hand_index]

        pixels = [
            (int(p[0] * width), int(p[1] * height))
            for p in points
        ]

        for a, b in connections:
            cv2.line(
                image, pixels[a], pixels[b],
                colors[hand_index], 2
            )

        for point in pixels:
            cv2.circle(
                image, point, 3,
                colors[hand_index], -1
            )

        wrist = pixels[0]

        cv2.putText(
            image,
            f"{prefix} {hand_index + 1}",
            (wrist[0], max(30, wrist[1] - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[hand_index],
            2
        )

    return image


cap = cv2.VideoCapture(VIDEO)

for frame_number in FRAMES:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame:", frame_number)
        continue

    original_view = draw_hands(
        frame,
        original_landmarks[frame_number],
        original_detected[frame_number],
        "Slot"
    )

    tracked_view = draw_hands(
        frame,
        tracked_landmarks[frame_number],
        tracked_detected[frame_number],
        "Track"
    )

    comparison = np.hstack([
        original_view,
        tracked_view
    ])

    cv2.putText(
        comparison,
        f"Frame {frame_number}: Original | Associated V2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    output_file = OUTPUT / f"comparison_{frame_number:03d}.jpg"

    cv2.imwrite(str(output_file), comparison)
    print("Saved:", output_file)

cap.release()

print("\nVisual comparison complete.")
