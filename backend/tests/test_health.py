from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.main import app


def test_health_returns_ok_and_correlation_id() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/health", headers={"X-Correlation-ID": "4c1575e0-965b-4e31-bdde-2c2dfe86ebcc"}
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Correlation-ID"] == "4c1575e0-965b-4e31-bdde-2c2dfe86ebcc"


def test_health_replaces_invalid_correlation_id() -> None:
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Correlation-ID": "not-a-uuid"})

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] != "not-a-uuid"


def test_readiness_checks_database_connection() -> None:
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_reports_database_unavailable() -> None:
    with patch(
        "app.api.routes.health.check_database_connection",
        side_effect=OperationalError("SELECT 1", {}, None),
    ):
        with TestClient(app) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "DATABASE_UNAVAILABLE",
        "message": "Database is not ready",
        "correlationId": response.headers["X-Correlation-ID"],
    }


def test_validation_error_uses_standard_error_shape_and_correlation_id() -> None:
    correlation_id = "4c1575e0-965b-4e31-bdde-2c2dfe86ebcc"
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/consultations",
            json={"expiresInMinutes": 0},
            headers={"X-Correlation-ID": correlation_id},
        )

    assert response.status_code == 422
    assert response.json() == {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed.",
        "correlationId": correlation_id,
        "fieldErrors": {"expiresInMinutes": ["Input should be greater than 0"]},
    }
