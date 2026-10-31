from uuid import UUID

import pytest

from app.inference import DeterministicMockRecognitionProvider, InferenceRequest
from app.inference.schemas import InferenceStatus, QualityStatus


def test_mock_provider_is_deterministic_and_abstains() -> None:
    provider = DeterministicMockRecognitionProvider()
    request = InferenceRequest(
        request_id=UUID("3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1"),
        duration_ms=2_000,
    )

    first_result = provider.recognize(request)
    second_result = provider.recognize(request)

    assert first_result == second_result
    assert first_result.provider == "deterministic-mock"
    assert first_result.model_version == "mock-v1"
    assert first_result.status is InferenceStatus.UNSUPPORTED
    assert first_result.quality_status is QualityStatus.NEEDS_CLARIFICATION
    assert first_result.candidates == []


def test_inference_request_rejects_out_of_range_duration() -> None:
    with pytest.raises(ValueError):
        InferenceRequest(
            request_id=UUID("3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1"),
            duration_ms=30_001,
        )


def test_inference_request_rejects_invalid_landmark_feature_dimensions() -> None:
    with pytest.raises(ValueError, match="258 finite values"):
        InferenceRequest(
            request_id=UUID("3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1"),
            duration_ms=2_000,
            features=((0.0,) * 257,),
        )


def test_inference_request_rejects_non_finite_features() -> None:
    with pytest.raises(ValueError):
        InferenceRequest(
            request_id=UUID("3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1"),
            duration_ms=2_000,
            features=((float("nan"),) + (0.0,) * 257,),
        )
