from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InferenceStatus(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    LOW_QUALITY = "LOW_QUALITY"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    CANDIDATES = "CANDIDATES"


class QualityStatus(StrEnum):
    ACCEPTABLE = "ACCEPTABLE"
    LOW_QUALITY = "LOW_QUALITY"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    duration_ms: int = Field(gt=0, le=30_000)


class InferenceCandidate(BaseModel):
    text: str
    score: float = Field(ge=0, le=1)
    evidence: list[str]


class InferenceResult(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda name: "".join(
            word.capitalize() if index else word for index, word in enumerate(name.split("_"))
        ),
        populate_by_name=True,
    )

    provider: str
    model_version: str
    status: InferenceStatus
    quality_status: QualityStatus
    recognized_glosses: list[str]
    candidates: list[InferenceCandidate]
    warnings: list[str]
    processing_time_ms: int = Field(ge=0)
