from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from app.dataset.models import (
    DATASET_VERSION_PATTERN,
    GLOSS_PATTERN,
    OPAQUE_REFERENCE_PATTERN,
)


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    status: Literal["PENDING_REVIEW", "APPROVED"]
    provider: Literal["temporal-baseline"]
    architecture: Literal["temporal-mean-delta-mlp"]
    weights_sha256: str = Field(alias="weightsSha256", pattern=r"^[a-f0-9]{64}$")
    dataset_version: str = Field(alias="datasetVersion", pattern=DATASET_VERSION_PATTERN)
    manifest_sha256: str = Field(alias="manifestSha256", pattern=r"^[a-f0-9]{64}$")
    split_report_sha256: str = Field(alias="splitReportSha256", pattern=r"^[a-f0-9]{64}$")
    feature_dimension: Literal[258] = Field(alias="featureDimension")
    class_to_index: dict[str, int] = Field(alias="classToIndex", min_length=2)
    independent_semantic_review: Literal["PENDING", "APPROVED"] = Field(
        alias="independentSemanticReview"
    )
    calibration_review: Literal["PENDING", "APPROVED"] = Field(alias="calibrationReview")
    approval_reference: str | None = Field(
        alias="approvalReference",
        default=None,
        pattern=OPAQUE_REFERENCE_PATTERN,
    )
    min_confidence: FiniteFloat | None = Field(alias="minConfidence", default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_class_map_and_approval(self) -> Self:
        if any(not re.fullmatch(GLOSS_PATTERN, gloss) for gloss in self.class_to_index):
            raise ValueError("model card classes must use canonical gloss labels")
        if set(self.class_to_index.values()) != set(range(len(self.class_to_index))):
            raise ValueError("model card class indexes must be contiguous and unique")
        if self.status == "APPROVED" and (
            self.independent_semantic_review != "APPROVED"
            or self.calibration_review != "APPROVED"
            or self.approval_reference is None
            or self.min_confidence is None
        ):
            raise ValueError(
                "approved model cards require semantic, calibration and threshold approval"
            )
        return self
