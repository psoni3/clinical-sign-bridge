import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.inference.video_predictions as video_predictions
import app.main as main_module
import extract_include_landmarks as landmark_extractor
from app.inference.video_predictions import (
    InvalidVideoError,
    VideoFrameLimitExceededError,
)


class FakeVideoPredictionService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.paths: list[Path] = []

    def predict_video(self, video_path: Path) -> tuple[int, list[tuple[str, float]]]:
        self.paths.append(video_path)
        assert video_path.is_file()
        if self.error is not None:
            raise self.error
        return 12, [
            ("SIGN-1", 0.7),
            ("SIGN-2", 0.1),
            ("SIGN-3", 0.08),
            ("SIGN-4", 0.07),
            ("SIGN-5", 0.05),
        ]


@contextmanager
def prediction_client(
    monkeypatch,
    service: FakeVideoPredictionService | None,
    *,
    provider_name: str = "bilstm-v1",
    max_upload_bytes: int = 50 * 1024 * 1024,
    max_video_frames: int = 250,
):
    build_calls: list[tuple[str, Path, int]] = []

    def build_service(name: str, artifact_dir: Path, frame_limit: int):
        build_calls.append((name, artifact_dir, frame_limit))
        return service if name == "bilstm-v1" else None

    monkeypatch.setenv("CLINICAL_VIDEO_PREDICTION_PROVIDER", provider_name)
    monkeypatch.setenv("CLINICAL_MAX_VIDEO_UPLOAD_BYTES", str(max_upload_bytes))
    monkeypatch.setenv("CLINICAL_MAX_VIDEO_FRAMES", str(max_video_frames))
    monkeypatch.setattr(main_module, "_build_video_prediction_service", build_service)

    with TestClient(main_module.app) as client:
        yield client, build_calls


def track_temporary_directories(monkeypatch) -> list[Path]:
    directories: list[Path] = []
    original_temporary_directory = video_predictions.TemporaryDirectory

    class TrackedTemporaryDirectory:
        def __init__(self, *args, **kwargs) -> None:
            self._temporary_directory = original_temporary_directory(*args, **kwargs)
            self.name = self._temporary_directory.name
            directories.append(Path(self.name))

        def cleanup(self) -> None:
            self._temporary_directory.cleanup()

    monkeypatch.setattr(
        video_predictions,
        "TemporaryDirectory",
        TrackedTemporaryDirectory,
    )
    return directories


def test_upload_prediction_loads_once_reuses_service_and_cleans_files(
    monkeypatch,
) -> None:
    service = FakeVideoPredictionService()

    with prediction_client(monkeypatch, service) as (client, build_calls):
        first_response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.MOV", b"video-data", "video/quicktime")},
        )
        second_response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
        )

        assert client.get("/ready").json()["status"] == "ready"

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    payload = first_response.json()
    assert payload["predictionStatus"] == "UNCONFIRMED"
    assert payload["modelVersion"] == "bilstm-v1"
    assert payload["scoreType"] == "uncalibrated_softmax"
    assert payload["frameCount"] == 12
    assert len(payload["candidates"]) == 5
    assert payload["candidates"][0] == {"label": "SIGN-1", "score": 0.7}
    assert "unconfirmed" in payload["warning"].lower()
    assert len(build_calls) == 1
    assert build_calls[0][2] == 250
    assert len(service.paths) == 2
    assert all(not path.exists() and not path.parent.exists() for path in service.paths)


def test_upload_rejects_unsupported_format(monkeypatch) -> None:
    service = FakeVideoPredictionService()

    with prediction_client(monkeypatch, service) as (client, _):
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.avi", b"video-data", "video/x-msvideo")},
        )

    assert response.status_code == 415
    assert service.paths == []


def test_upload_limit_is_checked_while_streaming_and_temp_storage_is_cleaned(
    monkeypatch,
) -> None:
    service = FakeVideoPredictionService()
    temporary_directories = track_temporary_directories(monkeypatch)

    with prediction_client(monkeypatch, service, max_upload_bytes=3) as (client, _):
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"four", "video/mp4")},
        )

    assert response.status_code == 413
    assert service.paths == []
    assert len(temporary_directories) == 1
    assert not temporary_directories[0].exists()


def test_empty_upload_is_rejected_and_temp_storage_is_cleaned(monkeypatch) -> None:
    service = FakeVideoPredictionService()
    temporary_directories = track_temporary_directories(monkeypatch)

    with prediction_client(monkeypatch, service) as (client, _):
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"", "video/mp4")},
        )

    assert response.status_code == 422
    assert service.paths == []
    assert not temporary_directories[0].exists()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (InvalidVideoError("private path"), 422),
        (VideoFrameLimitExceededError("private frame detail"), 422),
    ],
)
def test_video_validation_errors_are_safe_and_temp_storage_is_cleaned(
    monkeypatch,
    error,
    expected_status,
) -> None:
    service = FakeVideoPredictionService(error)

    with prediction_client(monkeypatch, service) as (client, _):
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
        )

    assert response.status_code == expected_status
    assert "private" not in response.text
    assert not service.paths[0].exists()
    assert not service.paths[0].parent.exists()


def test_unexpected_inference_failure_returns_generic_500_and_cleans_temp_files(
    monkeypatch,
) -> None:
    service = FakeVideoPredictionService(RuntimeError("private path and model details"))

    with prediction_client(monkeypatch, service) as (client, _):
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Video inference failed."
    assert "private" not in response.text
    assert not service.paths[0].exists()
    assert not service.paths[0].parent.exists()


def test_disabled_provider_returns_503_and_preserves_health_responses(monkeypatch) -> None:
    with prediction_client(
        monkeypatch,
        None,
        provider_name="disabled",
    ) as (client, build_calls):
        health = client.get("/health")
        readiness = client.get("/ready")
        response = client.post(
            "/api/v1/predictions",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert readiness.status_code == 200
    assert readiness.json() == {
        "status": "ready",
        "provider": main_module.configured_provider_name,
    }
    assert response.status_code == 503
    assert build_calls == []


def test_cors_allows_configured_patient_origin_and_rejects_unlisted_origin(
    monkeypatch,
) -> None:
    with prediction_client(
        monkeypatch,
        FakeVideoPredictionService(),
    ) as (client, _):
        preflight = client.options(
            "/api/v1/predictions",
            headers={
                "Origin": "http://localhost:4200",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-correlation-id",
            },
        )
        health = client.get(
            "/health",
            headers={
                "Origin": "http://localhost:4200",
                "X-Correlation-ID": "f6950a36-21ba-492a-91b5-fd9c4afaa53b",
            },
        )
        rejected = client.options(
            "/api/v1/predictions",
            headers={
                "Origin": "http://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:4200"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert "content-type" in preflight.headers["access-control-allow-headers"].lower()
    assert "*" not in preflight.headers["access-control-allow-methods"]
    assert health.headers["access-control-allow-origin"] == "http://localhost:4200"
    assert health.headers["x-correlation-id"] == "f6950a36-21ba-492a-91b5-fd9c4afaa53b"
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


def test_cors_origins_are_configurable_and_wildcards_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("CLINICAL_CORS_ORIGINS", '["https://patient.example"]')

    assert main_module._configured_cors_origins() == ["https://patient.example"]

    monkeypatch.setenv("CLINICAL_CORS_ORIGINS", '["*"]')
    with pytest.raises(ValueError, match="explicit HTTP\\(S\\) origins"):
        main_module._configured_cors_origins()


def test_video_provider_startup_failure_fails_application_startup(monkeypatch) -> None:
    def fail_startup(*_args):
        raise RuntimeError("model initialization failed")

    monkeypatch.setenv("CLINICAL_VIDEO_PREDICTION_PROVIDER", "bilstm-v1")
    monkeypatch.setattr(main_module, "_build_video_prediction_service", fail_startup)

    with pytest.raises(RuntimeError, match="model initialization failed"):
        with TestClient(main_module.app):
            pass


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        reported_frame_count: float = 0.0,
    ) -> None:
        self.frames = frames
        self.reported_frame_count = reported_frame_count
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_frame_count
        return 30.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeHolistic:
    def __init__(self) -> None:
        self.processed_frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def process(self, _frame):
        self.processed_frames += 1
        return type(
            "HolisticResult",
            (),
            {
                "pose_landmarks": None,
                "left_hand_landmarks": None,
                "right_hand_landmarks": None,
            },
        )()


def test_frame_limit_stops_feature_accumulation_during_decoding(
    monkeypatch,
    tmp_path,
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    capture = FakeCapture([frame.copy(), frame.copy(), frame.copy()])
    holistic = FakeHolistic()
    monkeypatch.setattr(landmark_extractor.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        landmark_extractor.mp_holistic,
        "Holistic",
        lambda **_kwargs: holistic,
    )

    with pytest.raises(landmark_extractor.VideoFrameLimitExceededError):
        landmark_extractor.process_video(
            tmp_path / "input.mp4",
            tmp_path / "features.npz",
            label="",
            relative_path="input.mp4",
            max_frames=2,
        )

    assert capture.read_count == 2
    assert capture.released
    assert holistic.processed_frames == 2
    assert not (tmp_path / "features.npz").exists()


def test_extractor_default_keeps_existing_unbounded_behavior(
    monkeypatch,
    tmp_path,
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    capture = FakeCapture([frame.copy(), frame.copy()])
    holistic = FakeHolistic()
    monkeypatch.setattr(landmark_extractor.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        landmark_extractor.mp_holistic,
        "Holistic",
        lambda **_kwargs: holistic,
    )
    output_path = tmp_path / "features.npz"

    result = landmark_extractor.process_video(
        tmp_path / "input.mp4",
        output_path,
        label="",
        relative_path="input.mp4",
    )

    assert result["frames"] == 2
    assert result["feature_shape"] == "(2, 75, 3)"
    assert output_path.is_file()
    assert capture.released


def test_bilstm_service_loads_artifacts_once_for_repeated_predictions(
    monkeypatch,
    tmp_path,
) -> None:
    import torch

    from app.inference.bilstm_v1_service import BilstmV1Service
    from clinicalsignbridge_ai.data.sequence_normalizer import SequenceNormalizer
    from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier

    labels = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
    model = BiLSTMClassifier(
        input_size=228,
        hidden_size=2,
        num_layers=1,
        num_classes=len(labels),
        dropout=0.1,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": 228,
            "hidden_size": 2,
            "num_layers": 1,
            "num_classes": len(labels),
            "dropout": 0.1,
            "class_labels": labels,
        },
        tmp_path / "best_model.pt",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({label: index for index, label in enumerate(labels)}),
        encoding="utf-8",
    )
    statistics = {
        f"{name}_{stat}": np.zeros(end - start, dtype=np.float32)
        if stat == "mean"
        else np.ones(end - start, dtype=np.float32)
        for name, (start, end, _) in SequenceNormalizer.GROUPS.items()
        for stat in ("mean", "std")
    }
    np.savez(tmp_path / "normalization.npz", **statistics)

    with patch("torch.load", wraps=torch.load) as load_checkpoint:
        service = BilstmV1Service(tmp_path, max_video_frames=250)
        monkeypatch.setattr(
            service,
            "_extract_sequence",
            lambda _path: (torch.ones((2, 228), dtype=torch.float32), 2),
        )

        first_result = service.predict_video(tmp_path / "first.mp4")
        second_result = service.predict_video(tmp_path / "second.mov")

    assert load_checkpoint.call_count == 1
    assert first_result[0] == second_result[0] == 2
    assert len(first_result[1]) == len(second_result[1]) == 5


def test_service_serializes_video_processing() -> None:
    from app.inference.bilstm_v1_service import BilstmV1Service

    second_attempting_lock = Event()
    first_inside = Event()
    allow_first_to_finish = Event()
    second_inside = Event()

    class ObservedLock:
        def __init__(self) -> None:
            self._lock = Lock()
            self._attempt_count = 0
            self._attempt_guard = Lock()

        def __enter__(self):
            with self._attempt_guard:
                self._attempt_count += 1
                if self._attempt_count == 2:
                    second_attempting_lock.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self._lock.release()

    service = object.__new__(BilstmV1Service)
    service._inference_lock = ObservedLock()

    def run_inference(path: Path) -> tuple[int, list[tuple[str, float]]]:
        if path.name == "first":
            first_inside.set()
            assert allow_first_to_finish.wait(timeout=5)
        else:
            second_inside.set()
        return 1, [("SIGN", 1.0)]

    service._predict_video_locked = run_inference
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.predict_video, Path("first"))
        assert first_inside.wait(timeout=5)
        second = executor.submit(service.predict_video, Path("second"))
        assert second_attempting_lock.wait(timeout=5)
        assert not second_inside.is_set()
        allow_first_to_finish.set()
        assert first.result(timeout=5) == (1, [("SIGN", 1.0)])
        assert second.result(timeout=5) == (1, [("SIGN", 1.0)])

    assert second_inside.is_set()
