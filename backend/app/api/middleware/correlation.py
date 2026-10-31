from uuid import UUID, uuid4

from fastapi import Request, Response
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = self._get_correlation_id(request.headers.get("X-Correlation-ID"))
        request.state.correlation_id = correlation_id
        trace.get_current_span().set_attribute("correlation_id", correlation_id)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @staticmethod
    def _get_correlation_id(value: str | None) -> str:
        try:
            return str(UUID(value)) if value else str(uuid4())
        except ValueError:
            return str(uuid4())
