import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_settings_reject_non_http_cors_origin() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url=SecretStr("sqlite+pysqlite:///:memory:"),
            cors_origins=["ftp://localhost:4200"],
        )
