from functools import lru_cache

from pydantic import AnyHttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: SecretStr
    cors_origins: list[str] = [
        "http://localhost:4200",
        "http://localhost:4201",
    ]
    otel_service_name: str = "clinical-sign-bridge-api"
    otel_exporter_otlp_endpoint: str | None = None

    model_config = SettingsConfigDict(env_prefix="CLINICAL_", env_file=".env", extra="ignore")

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        return [str(AnyHttpUrl(origin)).rstrip("/") for origin in origins]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
