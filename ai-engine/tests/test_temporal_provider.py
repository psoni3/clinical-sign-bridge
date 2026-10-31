import hashlib
import json

import pytest
from pydantic import ValidationError

from app.inference.model_card import ModelCard
from app.inference.temporal_provider import ModelConfigurationError, TemporalBaselineProvider


def _pending_card_payload(weights: bytes) -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "status": "PENDING_REVIEW",
        "provider": "temporal-baseline",
        "architecture": "temporal-mean-delta-mlp",
        "weightsSha256": hashlib.sha256(weights).hexdigest(),
        "datasetVersion": "0.1.0",
        "manifestSha256": "a" * 64,
        "splitReportSha256": "b" * 64,
        "featureDimension": 258,
        "classToIndex": {"PAIN": 0, "UNKNOWN": 1},
        "independentSemanticReview": "PENDING",
        "calibrationReview": "PENDING",
        "approvalReference": None,
        "minConfidence": None,
    }


def test_model_card_rejects_approval_without_independent_reviews() -> None:
    payload = _pending_card_payload(b"test-weights")
    payload.update(
        {
            "status": "APPROVED",
            "approvalReference": "approval_abc123",
            "minConfidence": 0.8,
        }
    )

    with pytest.raises(ValidationError, match="semantic, calibration"):
        ModelCard.model_validate(payload)


def test_temporal_provider_refuses_pending_model_before_loading_weights(tmp_path) -> None:
    weights = b"not-a-real-model"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.pt").write_bytes(weights)
    (model_dir / "model-card.json").write_text(
        json.dumps(_pending_card_payload(weights)),
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="has not been approved"):
        TemporalBaselineProvider(model_dir)
