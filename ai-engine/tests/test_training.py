import json

import pytest

from app.dataset.splits import assign_signer_separated_splits
from app.training.cli import TrainingDataError, _classification_metrics, _read_split_report


def test_training_split_loader_accepts_only_matching_signer_assignments(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    split = assign_signer_separated_splits(approved_dataset_manifest, seed=17)
    split_path = tmp_path / "split-report.json"
    split_path.write_text(split.to_json(), encoding="utf-8")

    assignments, digest = _read_split_report(split_path, approved_dataset_manifest)

    assert assignments == {
        str(sample_id): split_name for sample_id, split_name in split.sample_assignments.items()
    }
    assert len(digest) == 64


def test_training_split_loader_rejects_malformed_report_json(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    split_path = tmp_path / "split-report.json"
    split_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="unsupported schema"):
        _read_split_report(split_path, approved_dataset_manifest)


def test_training_split_loader_rejects_signer_leakage(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    split = assign_signer_separated_splits(approved_dataset_manifest, seed=17)
    report = json.loads(split.to_json())
    sample_id, split_name = next(iter(report["sampleAssignments"].items()))
    report["sampleAssignments"][sample_id] = "test" if split_name != "test" else "train"
    split_path = tmp_path / "split-report.json"
    split_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="assigns one signer to multiple splits"):
        _read_split_report(split_path, approved_dataset_manifest)


def test_training_split_loader_rejects_inconsistent_report_counts(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    split = assign_signer_separated_splits(approved_dataset_manifest, seed=17)
    report = json.loads(split.to_json())
    report["sampleCounts"]["train"] += 1
    split_path = tmp_path / "split-report.json"
    split_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="sample counts"):
        _read_split_report(split_path, approved_dataset_manifest)


def test_classification_metrics_report_per_class_errors_and_calibration() -> None:
    metrics = _classification_metrics(
        expected=[0, 1, 1],
        probabilities=[[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]],
        class_labels=["PAIN", "UNKNOWN"],
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macroF1"] == pytest.approx(2 / 3)
    assert metrics["brierScore"] == pytest.approx(0.46)
    assert metrics["confusionMatrix"] == [[1, 0], [1, 1]]
    assert metrics["perClass"]["UNKNOWN"]["support"] == 2
