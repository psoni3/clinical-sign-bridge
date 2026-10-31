import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_refuse_production_until_pilot_controls_are_approved() -> None:
    with pytest.raises(ValidationError, match="production deployment is disabled"):
        Settings(
            database_url="postgresql+psycopg://user:pass@localhost/db",
            deployment_mode="production",
        )


def test_settings_reject_wildcard_cors_origin() -> None:
    with pytest.raises(ValidationError, match="wildcard CORS origins"):
        Settings(database_url="sqlite+pysqlite:///:memory:", cors_origins=["*"])


def test_settings_reject_wildcard_trusted_host() -> None:
    with pytest.raises(ValidationError, match="explicit hostnames"):
        Settings(database_url="sqlite+pysqlite:///:memory:", trusted_hosts=["*"])
