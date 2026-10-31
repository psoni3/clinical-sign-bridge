import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.api.middleware.correlation import CorrelationIdMiddleware
from app.api.routes.confirmations import router as confirmations_router
from app.api.routes.consents import router as consents_router
from app.api.routes.consultations import router as consultations_router
from app.api.routes.health import router as health_router
from app.api.routes.interpretations import router as interpretations_router
from app.api.routes.messages import router as messages_router
from app.api.routes.recognitions import router as recognitions_router
from app.api.schemas import ErrorResponse
from app.api.websocket_handler import publish_outbox_delivery
from app.api.websocket_handler import router as websocket_router
from app.config import get_settings
from app.infrastructure.observability.tracing import configure_tracing
from app.infrastructure.persistence.outbox import create_outbox_dispatcher

settings = get_settings()
configure_tracing(
    settings.otel_service_name,
    str(settings.otel_exporter_otlp_endpoint) if settings.otel_exporter_otlp_endpoint else None,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    stop_event = asyncio.Event()
    dispatcher = create_outbox_dispatcher(
        settings.database_url.get_secret_value(),
        publish_outbox_delivery,
    )
    worker = asyncio.create_task(dispatcher.run(stop_event), name="notification-outbox")
    try:
        yield
    finally:
        stop_event.set()
        await worker


app = FastAPI(
    title="Clinical Sign Bridge API",
    description="ISL Healthcare Assistant - Consultation and Conversation Backend (M1)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
    detail: dict[str, object] = error.detail if isinstance(error.detail, dict) else {}
    payload = ErrorResponse(
        code=str(detail.get("code", "HTTP_ERROR")),
        message=str(detail.get("message", "The request could not be completed.")),
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return JSONResponse(
        status_code=error.status_code,
        content=payload.model_dump(mode="json", by_alias=True, exclude_none=True),
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    field_errors: dict[str, list[str]] = {}
    for item in error.errors():
        path = ".".join(
            str(part) for part in item["loc"] if part not in {"body", "query", "path", "header"}
        )
        field_errors.setdefault(path or "request", []).append(item["msg"])
    payload = ErrorResponse(
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        correlation_id=getattr(request.state, "correlation_id", None),
        field_errors=field_errors,
    )
    return JSONResponse(
        status_code=422,
        content=payload.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Correlation-ID"],
)


@app.middleware("http")
async def apply_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


FastAPIInstrumentor.instrument_app(app)
app.include_router(health_router)
app.include_router(consultations_router)
app.include_router(consents_router)
app.include_router(messages_router)
app.include_router(recognitions_router)
app.include_router(interpretations_router)
app.include_router(confirmations_router)
app.include_router(websocket_router)
