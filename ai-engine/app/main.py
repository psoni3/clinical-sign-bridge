import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.inference import (
    DeterministicMockRecognitionProvider,
    InferenceRequest,
    InferenceResult,
    RecognitionProvider,
)
from app.inference.temporal_provider import ModelConfigurationError, TemporalBaselineProvider

app = FastAPI(
    title="Clinical Sign Bridge Inference",
    description=(
        "Inference API. The deterministic mock is the safe default; research models are opt-in."
    ),
    version="0.1.0",
)
configured_provider_name = os.environ.get("CLINICAL_INFERENCE_PROVIDER", "deterministic-mock")


def _build_provider() -> RecognitionProvider:
    provider_name = configured_provider_name
    if provider_name == "deterministic-mock":
        return DeterministicMockRecognitionProvider()
    if provider_name == "temporal-baseline":
        model_dir = os.environ.get("CLINICAL_MODEL_DIR")
        if not model_dir:
            raise ModelConfigurationError(
                "CLINICAL_MODEL_DIR is required for the temporal-baseline provider"
            )
        return TemporalBaselineProvider(Path(model_dir))
    raise ModelConfigurationError("CLINICAL_INFERENCE_PROVIDER is not supported")


provider: RecognitionProvider = _build_provider()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = self._get_correlation_id(request.headers.get("X-Correlation-ID"))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @staticmethod
    def _get_correlation_id(value: str | None) -> str:
        try:
            return str(UUID(value)) if value else str(uuid4())
        except ValueError:
            return str(uuid4())


app.add_middleware(CorrelationIdMiddleware)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def readiness() -> dict[str, str]:
    return {"status": "ready", "provider": configured_provider_name}


@app.post("/api/v1/inference", response_model=InferenceResult, tags=["inference"])
def infer(request: InferenceRequest) -> InferenceResult:
    return provider.recognize(request)
