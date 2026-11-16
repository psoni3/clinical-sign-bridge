from __future__ import annotations

import json
from dataclasses import dataclass
from typing import BinaryIO, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PREDICTIONS_PATH = "/api/v1/predictions"
SupportedModelVersion = Literal["bilstm-v1"]


class AIInferenceError(Exception):
    """Base error for communication with the AI inference service."""


class AIInferenceUnavailableError(AIInferenceError):
    """Raised when the AI service cannot be reached or returns a server error."""


class AIInferenceHTTPError(AIInferenceError):
    """Raised when the AI service rejects a prediction request."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"AI inference service rejected the request with HTTP {status_code}.")


class AIInferenceResponseError(AIInferenceError):
    """Raised when the AI service response violates its prediction contract."""


class _CandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Candidate label must not be blank.")
        return value


class _PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prediction_status: Literal["UNCONFIRMED"] = Field(alias="predictionStatus")
    model_version: SupportedModelVersion = Field(alias="modelVersion")
    score_type: Literal["uncalibrated_softmax"] = Field(alias="scoreType")
    frame_count: int = Field(alias="frameCount", gt=0)
    candidates: list[_CandidatePayload] = Field(min_length=5, max_length=5)
    warning: str = Field(min_length=1)


@dataclass(frozen=True)
class AIInferenceCandidate:
    rank: int
    label: str
    score: float


@dataclass(frozen=True)
class AIInferenceResult:
    prediction_status: Literal["UNCONFIRMED"]
    model_version: str
    score_type: Literal["uncalibrated_softmax"]
    frame_count: int
    candidates: tuple[AIInferenceCandidate, ...]
    warning: str


class AIInferenceClient:
    """Uploads a video to the internal AI prediction service.

    An injected HTTP client is caller-owned and must be closed by its owner. Without an
    injected client, a short-lived client is opened and closed for each request.
    """

    def __init__(
        self,
        base_url: str,
        connection_timeout_seconds: float,
        response_timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._timeout = httpx.Timeout(
            connect=connection_timeout_seconds,
            read=response_timeout_seconds,
            write=response_timeout_seconds,
            pool=connection_timeout_seconds,
        )

    async def predict_video(
        self,
        video: BinaryIO,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> AIInferenceResult:
        files = {"video": (filename, video, content_type)}
        url = f"{self._base_url}{PREDICTIONS_PATH}"

        try:
            if self._http_client is not None:
                response = await self._http_client.post(url, files=files, timeout=self._timeout)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as http_client:
                    response = await http_client.post(url, files=files, timeout=self._timeout)
        except httpx.TimeoutException:
            raise AIInferenceUnavailableError("AI inference request timed out.") from None
        except httpx.TransportError:
            raise AIInferenceUnavailableError("AI inference service is unreachable.") from None

        if 400 <= response.status_code < 500:
            raise AIInferenceHTTPError(response.status_code)
        if response.status_code >= 500:
            raise AIInferenceUnavailableError(
                f"AI inference service returned HTTP {response.status_code}."
            )
        if response.is_redirect:
            raise AIInferenceResponseError("AI inference service returned an unexpected redirect.")

        try:
            payload = _PredictionPayload.model_validate(response.json())
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            raise AIInferenceResponseError(
                "AI inference service returned an invalid prediction response."
            ) from None

        candidates = tuple(
            AIInferenceCandidate(
                rank=rank,
                label=candidate.label,
                score=candidate.score,
            )
            for rank, candidate in enumerate(payload.candidates, start=1)
        )
        return AIInferenceResult(
            prediction_status=payload.prediction_status,
            model_version=payload.model_version,
            score_type=payload.score_type,
            frame_count=payload.frame_count,
            candidates=candidates,
            warning=payload.warning,
        )
