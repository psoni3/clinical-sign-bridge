from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.dataset.models import (
    AnnotationStatus,
    ConsentPurpose,
    ConsentStatus,
    DatasetManifest,
    DatasetSample,
    ReviewStatus,
)


@pytest.fixture
def approved_dataset_manifest() -> DatasetManifest:
    samples = [
        DatasetSample(
            sample_id=UUID(int=index + 1),
            media_reference=f"clips/{index + 1}.mp4",
            media_sha256=f"{index + 1:064x}",
            signer_pseudonym=f"signer_{index % 3 + 1:032x}",
            consent_reference=UUID(int=index + 101),
            consent_status=ConsentStatus.GRANTED,
            consented_purposes=[ConsentPurpose.MODEL_TRAINING],
            dataset_version="0.1.0",
            gloss="PAIN",
            segments=[{"start_ms": 0, "end_ms": 800, "gloss": "PAIN"}],
            annotator_pseudonyms=[f"annotator_{index + 1:032x}"],
            reviewer_pseudonyms=[f"annotator_{index + 1001:032x}"],
            annotation_status=AnnotationStatus.REVIEWED,
            quality_flags=[],
        )
        for index in range(6)
    ]
    return DatasetManifest(
        schema_version="1.0",
        dataset_version="0.1.0",
        created_at=datetime(2026, 10, 31, tzinfo=UTC),
        purpose=ConsentPurpose.MODEL_TRAINING,
        governance_status=ReviewStatus.APPROVED,
        license_review_status=ReviewStatus.APPROVED,
        annotation_guide_status=ReviewStatus.APPROVED,
        approval_reference="approval_abc123",
        samples=samples,
    )
