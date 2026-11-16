
from datetime import UTC, datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

MANIFEST_DIR = BASE_DIR / "data/manifests/frozen"
VIDEO_DIR = BASE_DIR / "data/raw/INCLUDE/videos"
OUTPUT_DIR = BASE_DIR / "artifacts/features/INCLUDE"
REPORT_DIR = BASE_DIR / "data/manifests/validation"

SPLITS = ["train", "validation", "test"]

mp_holistic = mp.solutions.holistic


class VideoReadError(RuntimeError):
    """Raised when a source video cannot be opened or decoded."""


class VideoFrameLimitExceededError(ValueError):
    """Raised when a source video exceeds a requested decoded-frame limit."""


def extract_landmarks(landmarks, expected_count):
    if landmarks is None:
        return np.zeros((expected_count, 3), dtype=np.float32)

    return np.asarray(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32
    )


def is_valid_feature_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with np.load(path, allow_pickle=False) as data:
            landmarks = data["landmarks"]
            pose = data["pose_detected"]
            left = data["left_hand_detected"]
            right = data["right_hand_detected"]

            return (
                landmarks.ndim == 3
                and landmarks.shape[1:] == (75, 3)
                and len(landmarks) > 0
                and pose.shape == (len(landmarks),)
                and left.shape == (len(landmarks),)
                and right.shape == (len(landmarks),)
                and np.isfinite(landmarks).all()
            )
    except Exception:
        return False


def process_video(video_path, output_path, label, relative_path, max_frames=None):
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive.")

    cap = cv2.VideoCapture(str(video_path))
    fps = 0.0
    reported_frame_count = 0.0
    all_frames = []
    pose_masks = []
    left_masks = []
    right_masks = []

    try:
        if not cap.isOpened():
            raise VideoReadError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if max_frames is not None:
            reported_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if reported_frame_count > max_frames:
                raise VideoFrameLimitExceededError(
                    f"Video exceeds the {max_frames}-frame limit."
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
                if max_frames is not None and len(all_frames) >= max_frames:
                    if reported_frame_count == max_frames:
                        break
                    raise VideoFrameLimitExceededError(
                        f"Video exceeds or cannot be verified against the "
                        f"{max_frames}-frame limit."
                    )

                success, frame = cap.read()

                if not success:
                    break

                if frame is None:
                    raise VideoReadError("Video decoder returned an empty frame")

                height, width = frame.shape[:2]

                if width > 640:
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
                left = extract_landmarks(
                    results.left_hand_landmarks, 21
                )
                right = extract_landmarks(
                    results.right_hand_landmarks, 21
                )

                frame_features = np.concatenate(
                    [pose, left, right], axis=0
                )

                all_frames.append(frame_features)

                pose_masks.append(
                    int(results.pose_landmarks is not None)
                )
                left_masks.append(
                    int(results.left_hand_landmarks is not None)
                )
                right_masks.append(
                    int(results.right_hand_landmarks is not None)
                )

    finally:
        cap.release()

    if not all_frames:
        raise VideoReadError("No frames extracted")

    landmarks_array = np.stack(all_frames).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        landmarks=landmarks_array,
        pose_detected=np.asarray(pose_masks, dtype=np.uint8),
        left_hand_detected=np.asarray(left_masks, dtype=np.uint8),
        right_hand_detected=np.asarray(right_masks, dtype=np.uint8),
        fps=np.float32(fps),
        label=np.asarray(label),
        source_video=np.asarray(relative_path)
    )

    return {
        "frames": len(all_frames),
        "fps": fps,
        "pose_rate": float(np.mean(pose_masks)),
        "left_rate": float(np.mean(left_masks)),
        "right_rate": float(np.mean(right_masks)),
        "feature_shape": str(landmarks_array.shape)
    }


def main():
    import pandas as pd

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for split in SPLITS:
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        manifest = pd.read_csv(manifest_path)

        print(f"\nProcessing {split}: {len(manifest)} videos")

        for index, row in manifest.iterrows():
            relative_path = str(row["video_path"]).replace("\\", "/")
            label = str(row["label"])

            video_path = VIDEO_DIR / relative_path

            # Preserve original path to avoid filename collisions.
            output_path = (
                OUTPUT_DIR / split / (relative_path + ".npz")
            )

            record = {
                "video_path": relative_path,
                "label": label,
                "split": split,
                "status": "",
                "frames": 0,
                "fps": 0,
                "pose_rate": 0,
                "left_hand_rate": 0,
                "right_hand_rate": 0,
                "feature_shape": "",
                "error": ""
            }

            try:
                if not video_path.is_file():
                    raise FileNotFoundError(str(video_path))

                if is_valid_feature_file(output_path):
                    with np.load(output_path, allow_pickle=False) as data:
                        landmarks = data["landmarks"]
                        record.update({
                            "status": "SKIPPED_VALID",
                            "frames": len(landmarks),
                            "fps": float(data["fps"]),
                            "pose_rate": float(data["pose_detected"].mean()),
                            "left_hand_rate": float(data["left_hand_detected"].mean()),
                            "right_hand_rate": float(data["right_hand_detected"].mean()),
                            "feature_shape": str(landmarks.shape)
                        })
                else:
                    stats = process_video(
                        video_path,
                        output_path,
                        label,
                        relative_path
                    )
                    record.update(stats)
                    record["status"] = "EXTRACTED"

            except Exception as exc:
                record["status"] = "FAILED"
                record["error"] = str(exc)

            results.append(record)

            processed = index + 1
            if processed % 10 == 0 or processed == len(manifest):
                print(
                    f"{split}: {processed}/{len(manifest)} "
                    f"| Status: {record['status']}"
                )

            # Save progress regularly, so the report survives interruption.
            if len(results) % 25 == 0:
                pd.DataFrame(results).to_csv(
                    REPORT_DIR / "landmark_extraction_progress.csv",
                    index=False
                )

    report = pd.DataFrame(results)

    report_path = REPORT_DIR / "landmark_extraction_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 60)
    print("LANDMARK EXTRACTION SUMMARY")
    print("=" * 60)
    print("Total videos:", len(report))
    print("\nStatus counts:")
    print(report["status"].value_counts().to_string())

    successful = report[
        report["status"].isin(["EXTRACTED", "SKIPPED_VALID"])
    ]

    print("\nAverage detection rates:")
    if not successful.empty:
        print("Pose:", round(successful["pose_rate"].mean() * 100, 2), "%")
        print("Left hand:", round(successful["left_hand_rate"].mean() * 100, 2), "%")
        print("Right hand:", round(successful["right_hand_rate"].mean() * 100, 2), "%")

    print("\nReport:", report_path)
    print("Completed:", datetime.now(UTC).isoformat())


if __name__ == "__main__":
    main()
