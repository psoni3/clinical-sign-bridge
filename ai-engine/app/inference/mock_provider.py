from app.inference.schemas import (
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    QualityStatus,
)


class DeterministicMockRecognitionProvider:
    provider_name = "deterministic-mock"
    model_version = "mock-v1"

    def recognize(self, _request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            provider=self.provider_name,
            model_version=self.model_version,
            status=InferenceStatus.UNSUPPORTED,
            quality_status=QualityStatus.NEEDS_CLARIFICATION,
            recognized_glosses=[],
            candidates=[],
            warnings=["Mock provider only; no video or Indian Sign Language is recognized."],
            processing_time_ms=0,
        )
