import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_settings_reject_non_http_cors_origin() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url=SecretStr("sqlite+pysqlite:///:memory:"),
            cors_origins=["ftp://localhost:4200"],
        )


def test_settings_accept_ai_client_configuration() -> None:
    settings = Settings(
        database_url=SecretStr("sqlite+pysqlite:///:memory:"),
        ai_service_base_url="http://ai-engine:8001",
        ai_connection_timeout_seconds=1.5,
        ai_response_timeout_seconds=45,
        recognition_processing_lease_seconds=240,
    )

    assert settings.ai_service_base_url == "http://ai-engine:8001"
    assert settings.ai_connection_timeout_seconds == 1.5
    assert settings.ai_response_timeout_seconds == 45
    assert settings.recognition_processing_lease_seconds == 240


def test_settings_default_ai_service_url_matches_local_ai_server() -> None:
    settings = Settings(database_url=SecretStr("sqlite+pysqlite:///:memory:"))

    assert settings.ai_service_base_url == "http://127.0.0.1:8001"
    assert settings.recognition_processing_lease_seconds == 180


@pytest.mark.parametrize(
    "values",
    [
        {"ai_service_base_url": "ftp://ai-engine:8001"},
        {"ai_service_base_url": "http://user:password@ai-engine:8001"},
        {"ai_connection_timeout_seconds": 0},
        {"ai_response_timeout_seconds": -1},
        {"recognition_processing_lease_seconds": 0},
        {"recognition_processing_lease_seconds": 3601},
    ],
)
def test_settings_reject_invalid_ai_client_configuration(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=SecretStr("sqlite+pysqlite:///:memory:"), **values)
