import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.inference import (
    DeterministicMockRecognitionProvider,
    InferenceRequest,
    InferenceResult,
    RecognitionProvider,
)
from app.inference.temporal_provider import ModelConfigurationError, TemporalBaselineProvider
from app.inference.video_predictions import (
    VideoPredictionService,
)
from app.inference.video_predictions import (
    router as video_predictions_router,
)

configured_provider_name = os.environ.get("CLINICAL_INFERENCE_PROVIDER", "deterministic-mock")
DEFAULT_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024
DEFAULT_VIDEO_FRAME_LIMIT = 250
DEFAULT_CORS_ORIGINS = ["http://localhost:4200", "http://localhost:4201"]
CORS_ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
CORS_ALLOWED_HEADERS = ["Content-Type", "X-Correlation-ID"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_cors_origins() -> list[str]:
    raw_origins = os.environ.get("CLINICAL_CORS_ORIGINS")
    if raw_origins is None:
        return DEFAULT_CORS_ORIGINS.copy()

    try:
        origins = json.loads(raw_origins)
    except json.JSONDecodeError as error:
        raise ValueError("CLINICAL_CORS_ORIGINS must be a JSON array.") from error

    if not isinstance(origins, list) or not origins:
        raise ValueError("CLINICAL_CORS_ORIGINS must contain explicit origins.")

    validated_origins: list[str] = []
    for origin in origins:
        if not isinstance(origin, str) or "*" in origin:
            raise ValueError("CLINICAL_CORS_ORIGINS must contain explicit HTTP(S) origins.")
        parsed_origin = urlsplit(origin)
        if (
            parsed_origin.scheme not in {"http", "https"}
            or not parsed_origin.hostname
            or parsed_origin.path not in {"", "/"}
            or parsed_origin.query
            or parsed_origin.fragment
            or parsed_origin.username
            or parsed_origin.password
        ):
            raise ValueError("CLINICAL_CORS_ORIGINS must contain explicit HTTP(S) origins.")
        validated_origins.append(origin.rstrip("/"))

    return validated_origins


CORS_ALLOWED_ORIGINS = _configured_cors_origins()


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


def _positive_int_setting(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ModelConfigurationError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ModelConfigurationError(f"{name} must be a positive integer")
    return parsed


def _build_video_prediction_service(
    provider_name: str,
    artifact_dir: Path,
    max_video_frames: int,
) -> VideoPredictionService | None:
    if provider_name == "disabled":
        return None
    if provider_name != "bilstm-v1":
        raise ModelConfigurationError(
            "CLINICAL_VIDEO_PREDICTION_PROVIDER must be 'disabled' or 'bilstm-v1'"
        )

    from app.inference.bilstm_v1_service import BilstmV1Service

    return BilstmV1Service(artifact_dir, max_video_frames)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    video_provider_name = os.environ.get(
        "CLINICAL_VIDEO_PREDICTION_PROVIDER",
        "disabled",
    )
    if video_provider_name not in {"disabled", "bilstm-v1"}:
        raise ModelConfigurationError(
            "CLINICAL_VIDEO_PREDICTION_PROVIDER must be 'disabled' or 'bilstm-v1'"
        )

    application.state.video_prediction_provider_name = video_provider_name
    application.state.video_prediction_service = None
    application.state.max_video_upload_bytes = DEFAULT_UPLOAD_LIMIT_BYTES
    if video_provider_name == "bilstm-v1":
        artifact_dir = Path(
            os.environ.get(
                "CLINICAL_BILSTM_ARTIFACT_DIR",
                str(PROJECT_ROOT / "artifacts" / "bilstm"),
            )
        )
        max_upload_bytes = _positive_int_setting(
            "CLINICAL_MAX_VIDEO_UPLOAD_BYTES",
            DEFAULT_UPLOAD_LIMIT_BYTES,
        )
        max_video_frames = _positive_int_setting(
            "CLINICAL_MAX_VIDEO_FRAMES",
            DEFAULT_VIDEO_FRAME_LIMIT,
        )
        application.state.max_video_upload_bytes = max_upload_bytes
        application.state.video_prediction_service = _build_video_prediction_service(
            video_provider_name,
            artifact_dir,
            max_video_frames,
        )

    try:
        yield
    finally:
        application.state.video_prediction_service = None


app = FastAPI(
    title="Clinical Sign Bridge Inference",
    description=(
        "Inference API. The deterministic mock is the safe default; research models are opt-in."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)
app.include_router(video_predictions_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", response_model=None, tags=["health"])
def readiness(request: Request) -> dict[str, str] | JSONResponse:
    video_provider_name = getattr(
        request.app.state,
        "video_prediction_provider_name",
        "disabled",
    )
    video_service = getattr(request.app.state, "video_prediction_service", None)
    if video_provider_name == "bilstm-v1" and video_service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "provider": configured_provider_name},
        )
    return {"status": "ready", "provider": configured_provider_name}


@app.post("/api/v1/inference", response_model=InferenceResult, tags=["inference"])
def infer(request: InferenceRequest) -> InferenceResult:
    return provider.recognize(request)
