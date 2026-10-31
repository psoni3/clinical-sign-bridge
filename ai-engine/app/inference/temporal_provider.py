from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.inference.model_card import ModelCard
from app.inference.schemas import (
    InferenceCandidate,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    QualityStatus,
)
from app.training.model import build_classifier, temporal_descriptor


class ModelConfigurationError(RuntimeError):
    """Raised when an opted-in inference model is missing valid approval artifacts."""


class TemporalBaselineProvider:
    def __init__(self, model_dir: Path):
        root = model_dir.resolve(strict=True)
        if not root.is_dir():
            raise ModelConfigurationError("configured model path must be a directory")
        card_path = root / "model-card.json"
        weights_path = root / "weights.pt"
        try:
            card_payload = json.loads(card_path.read_text(encoding="utf-8"))
            self.card = ModelCard.model_validate(card_payload)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ModelConfigurationError("configured model card is missing or invalid") from error
        if self.card.status != "APPROVED":
            raise ModelConfigurationError("configured model card has not been approved")
        if self.card.min_confidence is None:
            raise ModelConfigurationError("approved model card is missing its confidence threshold")
        resolved_weights_path = weights_path.resolve(strict=True)
        if not resolved_weights_path.is_relative_to(root) or not resolved_weights_path.is_file():
            raise ModelConfigurationError("model weights must be a file within the model directory")
        if self._sha256_file(resolved_weights_path) != self.card.weights_sha256:
            raise ModelConfigurationError("model weights checksum does not match the approved card")

        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ModelConfigurationError(
                "install the AI package training extra to load the temporal baseline"
            ) from error
        self.model = build_classifier(len(self.card.class_to_index))
        try:
            state = torch.load(resolved_weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            raise ModelConfigurationError("could not load approved model weights") from error
        self.model.eval()
        self.torch: Any = torch
        self.index_to_class = {index: gloss for gloss, index in self.card.class_to_index.items()}

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for block in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def recognize(self, request: InferenceRequest) -> InferenceResult:
        started = time.perf_counter()
        if request.features is None:
            return InferenceResult(
                provider=self.card.provider,
                model_version=self.card.dataset_version,
                status=InferenceStatus.UNSUPPORTED,
                quality_status=QualityStatus.NEEDS_CLARIFICATION,
                recognized_glosses=[],
                candidates=[],
                warnings=["landmark features are required; no recognition was attempted"],
                processing_time_ms=0,
            )

        sequence = self.torch.tensor(request.features, dtype=self.torch.float32).unsqueeze(0)
        with self.torch.no_grad():
            logits = self.model(temporal_descriptor(sequence))
            probabilities = self.torch.softmax(logits, dim=1).squeeze(0)
        class_index = int(probabilities.argmax().item())
        confidence = float(probabilities[class_index].item())
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        min_confidence = self.card.min_confidence
        if min_confidence is None:
            raise ModelConfigurationError("approved model card is missing its confidence threshold")
        if confidence < min_confidence:
            return InferenceResult(
                provider=self.card.provider,
                model_version=self.card.dataset_version,
                status=InferenceStatus.NEEDS_CLARIFICATION,
                quality_status=QualityStatus.NEEDS_CLARIFICATION,
                recognized_glosses=[],
                candidates=[],
                warnings=["model confidence is below the independently approved threshold"],
                processing_time_ms=elapsed_ms,
            )

        gloss = self.index_to_class[class_index]
        if gloss == "UNKNOWN":
            return InferenceResult(
                provider=self.card.provider,
                model_version=self.card.dataset_version,
                status=InferenceStatus.UNSUPPORTED,
                quality_status=QualityStatus.NEEDS_CLARIFICATION,
                recognized_glosses=[],
                candidates=[],
                warnings=["model selected the explicit unknown gloss class"],
                processing_time_ms=elapsed_ms,
            )
        return InferenceResult(
            provider=self.card.provider,
            model_version=self.card.dataset_version,
            status=InferenceStatus.CANDIDATES,
            quality_status=QualityStatus.ACCEPTABLE,
            recognized_glosses=[gloss],
            candidates=[
                InferenceCandidate(
                    text=gloss,
                    score=confidence,
                    evidence=["temporal-landmark-classifier"],
                )
            ],
            warnings=["isolated-gloss research model; not clinical interpretation"],
            processing_time_ms=elapsed_ms,
        )
