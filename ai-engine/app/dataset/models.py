from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_VERSION_PATTERN = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
OPAQUE_REFERENCE_PATTERN = r"^[a-z][a-z0-9_-]{2,63}$"
SIGNER_PATTERN = r"^signer_[0-9a-f]{32}$"
ANNOTATOR_PATTERN = r"^annotator_[0-9a-f]{32}$"
GLOSS_PATTERN = r"^[A-Z][A-Z0-9_]{0,79}$"
MEDIA_REFERENCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$"
REGIONAL_VARIANT_PATTERN = r"^[A-Z][A-Z0-9_-]{0,79}$"
QUALITY_FLAG_PATTERN = r"^[A-Z][A-Z0-9_-]{0,79}$"


class ReviewStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ConsentPurpose(StrEnum):
    LIVE_PROCESSING = "LIVE_PROCESSING"
    RESEARCH_COLLECTION = "RESEARCH_COLLECTION"
    MODEL_TRAINING = "MODEL_TRAINING"


class ConsentStatus(StrEnum):
    GRANTED = "GRANTED"
    WITHDRAWN = "WITHDRAWN"


class AnnotationStatus(StrEnum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    DISPUTED = "DISPUTED"


class AnnotationSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    gloss: str = Field(pattern=GLOSS_PATTERN)

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class DatasetSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: UUID
    media_reference: str = Field(pattern=MEDIA_REFERENCE_PATTERN)
    media_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    signer_pseudonym: str = Field(pattern=SIGNER_PATTERN)
    consent_reference: UUID
    consent_status: ConsentStatus
    consented_purposes: tuple[ConsentPurpose, ...] = Field(min_length=1)
    dataset_version: str = Field(pattern=DATASET_VERSION_PATTERN)
    gloss: str = Field(pattern=GLOSS_PATTERN)
    segments: tuple[AnnotationSegment, ...] = Field(min_length=1)
    regional_variant: str | None = Field(default=None, pattern=REGIONAL_VARIANT_PATTERN)
    annotator_pseudonyms: tuple[str, ...] = Field(min_length=1)
    reviewer_pseudonyms: tuple[str, ...] = ()
    annotation_status: AnnotationStatus
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_annotation_and_consent(self) -> Self:
        if len(set(self.annotator_pseudonyms)) != len(self.annotator_pseudonyms):
            raise ValueError("annotator_pseudonyms must be unique")
        if any(not re.fullmatch(ANNOTATOR_PATTERN, value) for value in self.annotator_pseudonyms):
            raise ValueError("annotator_pseudonyms must be opaque annotator references")
        if len(set(self.reviewer_pseudonyms)) != len(self.reviewer_pseudonyms):
            raise ValueError("reviewer_pseudonyms must be unique")
        if any(not re.fullmatch(ANNOTATOR_PATTERN, value) for value in self.reviewer_pseudonyms):
            raise ValueError("reviewer_pseudonyms must be opaque annotator references")
        if set(self.annotator_pseudonyms).intersection(self.reviewer_pseudonyms):
            raise ValueError("annotation reviewers must be independent of annotators")
        if any(not re.fullmatch(QUALITY_FLAG_PATTERN, flag) for flag in self.quality_flags):
            raise ValueError("quality_flags must be controlled uppercase labels")
        if self.consent_status is ConsentStatus.WITHDRAWN:
            raise ValueError("withdrawn samples must not be included in an active dataset version")
        if ConsentPurpose.MODEL_TRAINING not in self.consented_purposes:
            raise ValueError("dataset samples require separate model-training consent")
        if any(segment.end_ms > 30_000 for segment in self.segments):
            raise ValueError("annotation segments cannot exceed the 30-second capture limit")
        for preceding, following in zip(self.segments, self.segments[1:], strict=False):
            if following.start_ms < preceding.end_ms:
                raise ValueError("annotation segments must be ordered and non-overlapping")
        media_path = PurePosixPath(self.media_reference)
        if media_path.is_absolute() or ".." in media_path.parts:
            raise ValueError("media_reference must stay within the configured media root")
        if any(segment.gloss != self.gloss for segment in self.segments):
            raise ValueError("segment glosses must match the sample-level training label")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^1\.\d+$")
    dataset_version: str = Field(pattern=DATASET_VERSION_PATTERN)
    created_at: datetime
    purpose: ConsentPurpose
    governance_status: ReviewStatus
    license_review_status: ReviewStatus
    annotation_guide_status: ReviewStatus
    approval_reference: str | None = Field(default=None, pattern=OPAQUE_REFERENCE_PATTERN)
    samples: tuple[DatasetSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_version_and_sample_ids(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id values must be unique")
        if any(sample.dataset_version != self.dataset_version for sample in self.samples):
            raise ValueError("each sample dataset_version must match the manifest")
        return self

    def require_training_approval(self) -> None:
        if self.purpose is not ConsentPurpose.MODEL_TRAINING:
            raise ValueError("manifest purpose must be MODEL_TRAINING")
        if self.governance_status is not ReviewStatus.APPROVED:
            raise ValueError("dataset governance approval is required before training")
        if self.license_review_status is not ReviewStatus.APPROVED:
            raise ValueError("license review approval is required before training")
        if self.annotation_guide_status is not ReviewStatus.APPROVED:
            raise ValueError("annotation guide approval is required before training")
        if self.approval_reference is None:
            raise ValueError("an opaque dataset approval reference is required")
        for sample in self.samples:
            if sample.consent_status is not ConsentStatus.GRANTED:
                raise ValueError("every sample requires active, explicit consent")
            if ConsentPurpose.MODEL_TRAINING not in sample.consented_purposes:
                raise ValueError("every sample requires separate model-training consent")
            if sample.annotation_status is not AnnotationStatus.REVIEWED:
                raise ValueError("every sample requires completed annotation review")
            if not sample.reviewer_pseudonyms:
                raise ValueError("every sample requires an independent annotation reviewer")
