from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.api.middleware.correlation import CorrelationIdMiddleware
from app.api.routes.health import router as health_router
from app.config import get_settings
from app.infrastructure.observability.tracing import configure_tracing

settings = get_settings()
configure_tracing(
    settings.otel_service_name,
    str(settings.otel_exporter_otlp_endpoint) if settings.otel_exporter_otlp_endpoint else None,
)

app = FastAPI(
    title="Clinical Sign Bridge API",
    description="Prototype API foundation. M0 does not implement consultation workflows.",
    version="0.1.0",
)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Correlation-ID"],
)
FastAPIInstrumentor.instrument_app(app)
app.include_router(health_router)
