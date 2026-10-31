from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.inference import (
    DeterministicMockRecognitionProvider,
    InferenceRequest,
    InferenceResult,
    RecognitionProvider,
)

app = FastAPI(
    title="Clinical Sign Bridge Inference",
    description="Inference contract shell. The configured provider is a deterministic mock.",
    version="0.1.0",
)
provider: RecognitionProvider = DeterministicMockRecognitionProvider()


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
    return {"status": "ready", "provider": "deterministic-mock"}


@app.post("/api/v1/inference", response_model=InferenceResult, tags=["inference"])
def infer(request: InferenceRequest) -> InferenceResult:
    return provider.recognize(request)
