import pytest

from app.dataset.models import DatasetManifest
from app.dataset.splits import assign_signer_separated_splits


def test_split_assignment_is_deterministic_and_has_no_signer_leakage(
    approved_dataset_manifest,
) -> None:
    manifest = approved_dataset_manifest
    first = assign_signer_separated_splits(manifest, seed=17)
    second = assign_signer_separated_splits(manifest, seed=17)

    assert first == second
    for sample in manifest.samples:
        assert (
            first.sample_assignments[sample.sample_id]
            == first.signer_assignments[sample.signer_pseudonym]
        )
    assert set(first.signer_counts.values()) == {1}
    assert sum(first.sample_counts.values()) == len(manifest.samples)


def test_split_assignment_requires_training_approval(approved_dataset_manifest) -> None:
    manifest_data = approved_dataset_manifest.model_dump(mode="json")
    manifest_data["governance_status"] = "PENDING"
    manifest = DatasetManifest.model_validate(manifest_data)

    with pytest.raises(ValueError, match="governance approval"):
        assign_signer_separated_splits(manifest, seed=17)


def test_split_assignment_requires_at_least_three_signers(approved_dataset_manifest) -> None:
    manifest = type(approved_dataset_manifest).model_validate(
        approved_dataset_manifest.model_dump(mode="json")
    )
    manifest_data = manifest.model_dump(mode="json")
    manifest_data["samples"] = manifest_data["samples"][:2]
    manifest = DatasetManifest.model_validate(manifest_data)

    with pytest.raises(ValueError, match="at least three distinct signers"):
        assign_signer_separated_splits(manifest, seed=17)


def test_split_report_serializes_only_opaque_dataset_references(approved_dataset_manifest) -> None:
    split = assign_signer_separated_splits(approved_dataset_manifest, seed=17)
    report = split.to_json()

    assert "sampleAssignments" in report
    assert "signerAssignments" in report
    assert "patient name" not in report.lower()
