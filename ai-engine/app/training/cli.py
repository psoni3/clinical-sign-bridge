from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.dataset.manifest import DatasetManifestError, load_manifest, manifest_sha256
from app.dataset.models import DatasetManifest
from app.training.model import build_classifier, temporal_descriptor

MAX_EPOCHS = 500
MAX_SEQUENCE_FRAMES = 300


class TrainingDataError(ValueError):
    """Raised when dataset feature or signer split artifacts are inconsistent."""


def _read_split_report(path: Path, manifest: DatasetManifest) -> tuple[dict[str, str], str]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError("could not read signer-separated split report") from error
    if not isinstance(report, dict) or report.get("schemaVersion") != "1.0":
        raise TrainingDataError("split report has an unsupported schema")
    if report.get("datasetVersion") != manifest.dataset_version:
        raise TrainingDataError("split report dataset version does not match the manifest")
    expected_manifest_digest = manifest_sha256(manifest)
    if report.get("manifestSha256") != expected_manifest_digest:
        raise TrainingDataError("split report does not match the dataset manifest digest")
    assignments = report.get("sampleAssignments")
    signer_assignments = report.get("signerAssignments")
    if not isinstance(assignments, dict) or not isinstance(signer_assignments, dict):
        raise TrainingDataError("split report is missing sample or signer assignments")

    expected_ids = {str(sample.sample_id) for sample in manifest.samples}
    if set(assignments) != expected_ids:
        raise TrainingDataError("split report sample IDs do not match the dataset manifest")
    split_names = {"train", "validation", "test"}
    if any(
        not isinstance(value, str) or value not in split_names for value in assignments.values()
    ):
        raise TrainingDataError("split report contains an unsupported split name")
    expected_signers = {sample.signer_pseudonym for sample in manifest.samples}
    if set(signer_assignments) != expected_signers:
        raise TrainingDataError("split report signer IDs do not match the dataset manifest")
    for sample in manifest.samples:
        if signer_assignments.get(sample.signer_pseudonym) != assignments[str(sample.sample_id)]:
            raise TrainingDataError("split report assigns one signer to multiple splits")
    if (
        any(
            not isinstance(value, str) or value not in split_names
            for value in signer_assignments.values()
        )
        or set(signer_assignments.values()) != split_names
    ):
        raise TrainingDataError("train, validation and test splits must each contain a signer")
    expected_sample_counts = {
        split_name: sum(value == split_name for value in assignments.values())
        for split_name in split_names
    }
    expected_signer_counts = {
        split_name: sum(value == split_name for value in signer_assignments.values())
        for split_name in split_names
    }
    if report.get("sampleCounts") != expected_sample_counts:
        raise TrainingDataError("split report sample counts do not match its assignments")
    if report.get("signerCounts") != expected_signer_counts:
        raise TrainingDataError("split report signer counts do not match its assignments")

    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return assignments, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_features(
    index_path: Path,
    feature_dir: Path,
    manifest: DatasetManifest,
) -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError("could not read preprocessed feature index") from error

    if not isinstance(index, dict) or index.get("schemaVersion") != "1.0":
        raise TrainingDataError("feature index has an unsupported schema")
    if index.get("datasetVersion") != manifest.dataset_version:
        raise TrainingDataError("feature index dataset version does not match the manifest")
    if index.get("manifestSha256") != manifest_sha256(manifest):
        raise TrainingDataError("feature index does not match the dataset manifest digest")
    if index.get("featureDimension") != 258:
        raise TrainingDataError("feature index has an unsupported feature dimension")
    records = index.get("samples")
    expected_ids = {str(sample.sample_id) for sample in manifest.samples}
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise TrainingDataError("feature index contains invalid sample records")
    record_ids = [record.get("sampleId") for record in records]
    if any(not isinstance(sample_id, str) for sample_id in record_ids):
        raise TrainingDataError("feature index contains an invalid sample ID")
    if len(record_ids) != len(expected_ids) or set(record_ids) != expected_ids:
        raise TrainingDataError("feature index sample IDs do not match the dataset manifest")

    root = feature_dir.resolve(strict=True)
    samples_by_id = {str(sample.sample_id): sample for sample in manifest.samples}
    features_by_id: dict[str, Any] = {}
    for record in records:
        sample_id = record["sampleId"]
        if sample_id not in samples_by_id:
            raise TrainingDataError("feature index contains an invalid sample ID")
        sample = samples_by_id[sample_id]
        if record.get("gloss") != sample.gloss:
            raise TrainingDataError("feature index gloss does not match the dataset manifest")
        feature_name = record.get("featureFile")
        if not isinstance(feature_name, str) or feature_name != f"{sample_id}.npz":
            raise TrainingDataError("feature index contains an invalid feature file reference")
        feature_path = (root / feature_name).resolve(strict=True)
        if not feature_path.is_relative_to(root) or not feature_path.is_file():
            raise TrainingDataError("feature file must remain inside the feature directory")
        if record.get("featureSha256") != _sha256_file(feature_path):
            raise TrainingDataError("feature file checksum does not match the feature index")
        try:
            with numpy.load(feature_path, allow_pickle=False) as archive:
                features = archive["features"]
        except (OSError, ValueError, KeyError) as error:
            raise TrainingDataError("could not safely load a landmark feature file") from error
        if (
            features.ndim != 2
            or features.shape[1] != 258
            or not 1 <= features.shape[0] <= MAX_SEQUENCE_FRAMES
            or not numpy.issubdtype(features.dtype, numpy.number)
            or numpy.issubdtype(features.dtype, numpy.complexfloating)
            or not numpy.isfinite(features).all()
        ):
            raise TrainingDataError(f"invalid landmark feature shape for sample {sample_id}")
        with numpy.errstate(over="ignore", invalid="ignore"):
            normalized_features = features.astype(numpy.float32, copy=False)
        if not numpy.isfinite(normalized_features).all():
            raise TrainingDataError(f"invalid landmark feature values for sample {sample_id}")
        features_by_id[sample_id] = normalized_features
    return features_by_id


def _classification_metrics(
    expected: list[int],
    probabilities: list[list[float]],
    class_labels: list[str],
) -> dict[str, Any]:
    if not expected or len(expected) != len(probabilities):
        raise ValueError("classification metrics require matching non-empty inputs")
    class_count = len(class_labels)
    if any(not 0 <= label < class_count for label in expected):
        raise ValueError("classification labels are outside the configured class map")
    if any(
        len(probabilities_for_sample) != class_count for probabilities_for_sample in probabilities
    ):
        raise ValueError("probability vectors do not match the configured class map")
    predicted = [
        max(range(class_count), key=probabilities_for_sample.__getitem__)
        for probabilities_for_sample in probabilities
    ]
    correct = sum(actual == guess for actual, guess in zip(expected, predicted, strict=True))
    per_class: list[float] = []
    class_metrics: dict[str, dict[str, float | int]] = {}
    confusion_matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, guess in zip(expected, predicted, strict=True):
        confusion_matrix[actual][guess] += 1

    for label in range(class_count):
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        support = sum(actual == label for actual in expected)
        precision_denominator = true_positive + false_positive
        class_metrics[class_labels[label]] = {
            "precision": true_positive / precision_denominator if precision_denominator else 0.0,
            "recall": true_positive / support if support else 0.0,
            "f1": f1,
            "support": support,
        }
        per_class.append(f1)

    confidence_pairs = [
        (max(sample_probabilities), guess == actual)
        for sample_probabilities, guess, actual in zip(
            probabilities,
            predicted,
            expected,
            strict=True,
        )
    ]
    expected_calibration_error = 0.0
    for bin_index in range(10):
        low = bin_index / 10
        high = (bin_index + 1) / 10
        members = [
            pair
            for pair in confidence_pairs
            if low <= pair[0] < high or (bin_index == 9 and pair[0] == 1.0)
        ]
        if members:
            average_confidence = sum(confidence for confidence, _ in members) / len(members)
            average_accuracy = sum(float(is_correct) for _, is_correct in members) / len(members)
            expected_calibration_error += (
                len(members) / len(expected) * abs(average_accuracy - average_confidence)
            )
    brier_score = sum(
        sum(
            (probability - float(index == actual)) ** 2
            for index, probability in enumerate(sample_probabilities)
        )
        for actual, sample_probabilities in zip(expected, probabilities, strict=True)
    ) / len(expected)
    return {
        "accuracy": correct / len(expected),
        "macroF1": sum(per_class) / len(per_class),
        "expectedCalibrationError": expected_calibration_error,
        "brierScore": brier_score,
        "perClass": class_metrics,
        "confusionMatrix": confusion_matrix,
    }


def _evaluate(
    model: Any,
    records: list[tuple[Any, int]],
    class_labels: list[str],
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    model.eval()
    expected: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for features, label in records:
            sequence = torch.from_numpy(features).unsqueeze(0)
            logits = model(temporal_descriptor(sequence))
            probabilities.append(
                [float(value) for value in torch.softmax(logits, dim=1).squeeze(0).tolist()]
            )
            expected.append(label)
    return _classification_metrics(expected, probabilities, class_labels)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_baseline(
    manifest_path: Path,
    split_path: Path,
    feature_index_path: Path,
    feature_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    epochs: int,
) -> Path:
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    manifest = load_manifest(manifest_path)
    manifest.require_training_approval()
    assignments, split_digest = _read_split_report(split_path, manifest)
    features_by_id = _load_features(feature_index_path, feature_dir, manifest)

    classes = sorted({sample.gloss for sample in manifest.samples})
    if len(classes) < 2:
        raise TrainingDataError("the temporal baseline requires at least two gloss classes")
    class_to_index = {gloss: index for index, gloss in enumerate(classes)}
    split_records: dict[str, list[tuple[Any, int]]] = {
        split_name: [] for split_name in ("train", "validation", "test")
    }
    for sample in manifest.samples:
        split_records[assignments[str(sample.sample_id)]].append(
            (features_by_id[str(sample.sample_id)], class_to_index[sample.gloss])
        )
    if any(not split_records[name] for name in split_records):
        raise TrainingDataError("train, validation and test splits must each contain samples")
    train_labels = {label for _, label in split_records["train"]}
    if train_labels != set(range(len(classes))):
        raise TrainingDataError(
            "every evaluated gloss class must be represented in the training split"
        )

    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("install the AI package training extra to train a baseline") from error
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_classifier(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(split_records["train"]), generator=generator).tolist()
        for record_index in order:
            features, label = split_records["train"][record_index]
            sequence = torch.from_numpy(features).unsqueeze(0)
            target = torch.tensor([label], dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_function(model(temporal_descriptor(sequence)), target)
            loss.backward()
            optimizer.step()

    evaluation = {
        name: _evaluate(model, records, classes)
        for name, records in split_records.items()
        if name != "train"
    }
    if output_dir.exists() or output_dir.is_symlink():
        raise TrainingDataError("model output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        weights_path = staging_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        card = {
            "schemaVersion": "1.0",
            "status": "PENDING_REVIEW",
            "provider": "temporal-baseline",
            "architecture": "temporal-mean-delta-mlp",
            "weightsSha256": _sha256_file(weights_path),
            "datasetVersion": manifest.dataset_version,
            "manifestSha256": manifest_sha256(manifest),
            "splitReportSha256": split_digest,
            "featureDimension": 258,
            "classToIndex": class_to_index,
            "seed": seed,
            "epochs": epochs,
            "trainedAt": datetime.now(UTC).isoformat(),
            "independentSemanticReview": "PENDING",
            "calibrationReview": "PENDING",
            "approvalReference": None,
            "minConfidence": None,
        }
        (staging_dir / "model-card.json").write_text(
            json.dumps(card, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schemaVersion": "1.0",
            "datasetVersion": manifest.dataset_version,
            "manifestSha256": manifest_sha256(manifest),
            "splitReportSha256": split_digest,
            "sampleCounts": {name: len(records) for name, records in split_records.items()},
            "metrics": evaluation,
            "interpretation": "Research baseline metrics only; not clinical validation.",
        }
        (staging_dir / "evaluation.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
        return output_dir / "model-card.json"
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.training",
        description="Train and evaluate a governed isolated-gloss temporal baseline.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        model_card = train_baseline(
            arguments.manifest,
            arguments.split_report,
            arguments.feature_index,
            arguments.feature_dir,
            arguments.output_dir,
            seed=arguments.seed,
            epochs=arguments.epochs,
        )
        print(f"Created unapproved temporal baseline model card: {model_card}")
        return 0
    except (
        DatasetManifestError,
        OSError,
        TrainingDataError,
        ValidationError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"Training command failed: {error}", file=sys.stderr)
        return 2
