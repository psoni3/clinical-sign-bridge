from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing(service_name: str, exporter_endpoint: str | None) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=exporter_endpoint))
        )
    trace.set_tracer_provider(provider)
