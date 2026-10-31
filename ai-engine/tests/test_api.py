from fastapi.testclient import TestClient

from app.main import app


def test_inference_api_returns_explicit_mock_abstention() -> None:
    correlation_id = "4c1575e0-965b-4e31-bdde-2c2dfe86ebcc"
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference",
            headers={"X-Correlation-ID": correlation_id},
            json={
                "request_id": "3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1",
                "duration_ms": 2_000,
            },
        )

    assert response.status_code == 200
    assert response.json()["provider"] == "deterministic-mock"
    assert response.json()["modelVersion"] == "mock-v1"
    assert response.json()["status"] == "UNSUPPORTED"
    assert response.json()["candidates"] == []
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_inference_api_rejects_invalid_duration() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference",
            json={
                "request_id": "3b4299d5-1453-4f4b-9c0d-e8bb26fa04d1",
                "duration_ms": 0,
            },
        )

    assert response.status_code == 422
