from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.dataset.manifest import manifest_sha256
from app.dataset.models import DatasetManifest, DatasetSample

FEATURE_DIMENSION = 258


class PreprocessingError(ValueError):
    """Raised when a source clip cannot safely produce landmark features."""


def _resolve_media_file(media_root: Path, sample: DatasetSample) -> Path:
    root = media_root.resolve(strict=True)
    if not root.is_dir():
        raise PreprocessingError("media root must be a directory")
    media_path = (root / sample.media_reference).resolve(strict=True)
    if not media_path.is_relative_to(root) or not media_path.is_file():
        raise PreprocessingError("media reference must resolve to a file within the media root")

    digest = hashlib.sha256()
    try:
        with media_path.open("rb") as media_file:
            for block in iter(lambda: media_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreprocessingError("could not read media referenced by the manifest") from error
    if digest.hexdigest() != sample.media_sha256:
        raise PreprocessingError(f"media checksum mismatch for sample {sample.sample_id}")
    return media_path


def _landmark_values(landmarks: Any, point_count: int, include_visibility: bool) -> list[float]:
    if landmarks is None:
        values_per_point = 4 if include_visibility else 3
        return [0.0] * (point_count * values_per_point)
    values: list[float] = []
    for point in landmarks.landmark:
        values.extend((float(point.x), float(point.y), float(point.z)))
        if include_visibility:
            values.append(float(point.visibility))
    return values


def _extract_landmarks(
    media_path: Path,
    target_fps: int,
    max_frames: int,
) -> Any:
    try:
        cv2 = importlib.import_module("cv2")
        mediapipe = importlib.import_module("mediapipe")
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise PreprocessingError(
            "install the AI package preprocessing extra to extract landmarks"
        ) from error

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        capture.release()
        raise PreprocessingError("could not open a manifest media file")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, round(fps / target_fps)) if fps > 0 else 1
    frames: list[list[float]] = []
    frame_index = 0
    detected_hand = False
    try:
        with mediapipe.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            while len(frames) < max_frames:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb_frame)
                    detected_hand = detected_hand or result.left_hand_landmarks is not None
                    detected_hand = detected_hand or result.right_hand_landmarks is not None
                    values = _landmark_values(result.pose_landmarks, 33, True)
                    values.extend(_landmark_values(result.left_hand_landmarks, 21, False))
                    values.extend(_landmark_values(result.right_hand_landmarks, 21, False))
                    frames.append(values)
                frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise PreprocessingError("media file contains no decodable frames")
    if not detected_hand:
        raise PreprocessingError("no hand landmarks were detected in the source clip")

    features = numpy.asarray(frames, dtype=numpy.float32)
    if features.shape[1] != FEATURE_DIMENSION or not numpy.isfinite(features).all():
        raise PreprocessingError("landmark extraction returned invalid feature values")
    return features


def _write_features(path: Path, features: Any) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            importlib.import_module("numpy").savez_compressed(temporary_file, features=features)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as feature_file:
        for block in iter(lambda: feature_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_manifest(
    manifest: DatasetManifest,
    media_root: Path,
    output_dir: Path,
    *,
    target_fps: int = 15,
    max_frames: int = 96,
) -> Path:
    manifest.require_training_approval()
    if not 1 <= target_fps <= 30:
        raise ValueError("target_fps must be between 1 and 30")
    if not 1 <= max_frames <= 300:
        raise ValueError("max_frames must be between 1 and 300")

    if output_dir.exists() or output_dir.is_symlink():
        raise PreprocessingError("feature output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        records: list[dict[str, str]] = []
        for sample in manifest.samples:
            media_path = _resolve_media_file(media_root, sample)
            features_path = staging_dir / f"{sample.sample_id}.npz"
            features = _extract_landmarks(media_path, target_fps, max_frames)
            _write_features(features_path, features)
            records.append(
                {
                    "sampleId": str(sample.sample_id),
                    "featureFile": features_path.name,
                    "featureSha256": _sha256_file(features_path),
                    "gloss": sample.gloss,
                }
            )

        index = {
            "schemaVersion": "1.0",
            "datasetVersion": manifest.dataset_version,
            "manifestSha256": manifest_sha256(manifest),
            "featureDimension": FEATURE_DIMENSION,
            "targetFps": target_fps,
            "maxFrames": max_frames,
            "samples": records,
        }
        index_path = staging_dir / "index.json"
        index_path.write_text(
            json.dumps(index, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
        return output_dir / "index.json"
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
