from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: SecretStr
    cors_origins: list[str] = [
        "http://localhost:4200",
        "http://localhost:4201",
    ]
    trusted_hosts: list[str] = ["localhost", "127.0.0.1", "api", "testserver"]
    deployment_mode: Literal["local", "production"] = "local"
    otel_service_name: str = "clinical-sign-bridge-api"
    otel_exporter_otlp_endpoint: str | None = None
    ai_service_base_url: str = "http://127.0.0.1:8001"
    ai_connection_timeout_seconds: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    ai_response_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    recognition_processing_lease_seconds: int = Field(default=180, gt=0, le=3600)

    model_config = SettingsConfigDict(env_prefix="CLINICAL_", env_file=".env", extra="ignore")

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        if "*" in origins:
            raise ValueError("wildcard CORS origins are not allowed")
        return [str(AnyHttpUrl(origin)).rstrip("/") for origin in origins]

    @field_validator("trusted_hosts")
    @classmethod
    def validate_trusted_hosts(cls, hosts: list[str]) -> list[str]:
        if not hosts or "*" in hosts or any("/" in host or not host.strip() for host in hosts):
            raise ValueError("trusted_hosts must contain explicit hostnames without wildcards")
        return [host.strip().lower() for host in hosts]

    @field_validator("ai_service_base_url")
    @classmethod
    def validate_ai_service_base_url(cls, value: str) -> str:
        url = AnyHttpUrl(value)
        if url.username is not None or url.password is not None:
            raise ValueError("ai_service_base_url must not contain credentials")
        if url.query is not None or url.fragment is not None:
            raise ValueError("ai_service_base_url must not contain a query or fragment")
        return str(url).rstrip("/")

    @model_validator(mode="after")
    def prevent_unapproved_production_deployment(self) -> Settings:
        if self.deployment_mode == "production":
            raise ValueError(
                "production deployment is disabled until authentication/authorization, "
                "TLS termination, managed encryption, and required stakeholder approvals "
                "are implemented"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
