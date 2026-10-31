from typing import Protocol

from app.inference.schemas import InferenceRequest, InferenceResult


class RecognitionProvider(Protocol):
    def recognize(self, request: InferenceRequest) -> InferenceResult: ...
