import pytest
from pydantic import ValidationError

from app.dataset.manifest import manifest_sha256
from app.dataset.models import DatasetSample


def test_training_manifest_requires_explicit_purpose_specific_consent(
    approved_dataset_manifest,
) -> None:
    sample_data = approved_dataset_manifest.samples[0].model_dump(mode="json")
    sample_data["consented_purposes"] = ["LIVE_PROCESSING"]

    with pytest.raises(ValidationError, match="separate model-training consent"):
        DatasetSample.model_validate(sample_data)


def test_withdrawn_sample_cannot_be_reintroduced_into_active_manifest(
    approved_dataset_manifest,
) -> None:
    sample_data = approved_dataset_manifest.samples[0].model_dump(mode="json")
    sample_data["consent_status"] = "WITHDRAWN"

    with pytest.raises(ValidationError, match="withdrawn samples"):
        DatasetSample.model_validate(sample_data)


def test_manifest_digest_is_stable_for_equivalent_data(approved_dataset_manifest) -> None:
    second = type(approved_dataset_manifest).model_validate_json(
        approved_dataset_manifest.model_dump_json()
    )

    assert manifest_sha256(approved_dataset_manifest) == manifest_sha256(second)


def test_manifest_rejects_media_path_traversal(approved_dataset_manifest) -> None:
    sample_data = approved_dataset_manifest.samples[0].model_dump(mode="json")
    sample_data["media_reference"] = "../private/clip.mp4"

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(sample_data)


def test_manifest_does_not_accept_unreviewed_data_as_training_ready(
    approved_dataset_manifest,
) -> None:
    manifest_data = approved_dataset_manifest.model_dump(mode="json")
    manifest_data["samples"][0]["reviewer_pseudonyms"] = []
    manifest = type(approved_dataset_manifest).model_validate(manifest_data)

    with pytest.raises(ValueError, match="independent annotation reviewer"):
        manifest.require_training_approval()
