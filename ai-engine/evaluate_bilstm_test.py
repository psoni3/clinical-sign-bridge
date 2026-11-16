
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinicalsignbridge_ai.data.sequence_dataset import SignSequenceDataset
from clinicalsignbridge_ai.data.sequence_collator import collate_sign_sequences
from clinicalsignbridge_ai.data.sequence_normalizer import SequenceNormalizer
from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier


FEATURE_DIR = PROJECT_ROOT / "artifacts" / "features" / "INCLUDE"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "frozen"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "bilstm"
OUTPUT_DIR = MODEL_DIR / "test_evaluation"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16


class NormalizedDataset(Dataset):
    def __init__(self, base, normalizer):
        self.base = base
        self.normalizer = normalizer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sequence, label, length = self.base[index]
        return self.normalizer.transform(sequence), label, length


def load_normalizer(path):
    normalizer = SequenceNormalizer()

    with np.load(path, allow_pickle=False) as data:
        for name in normalizer.GROUPS:
            normalizer.statistics[name] = {
                "mean": data[f"{name}_mean"].copy(),
                "std": data[f"{name}_std"].copy(),
            }

    normalizer.fitted = True
    return normalizer


def main():
    checkpoint_path = MODEL_DIR / "best_model.pt"
    normalizer_path = MODEL_DIR / "normalization.npz"
    manifest_path = MANIFEST_DIR / "test.csv"

    for path in [checkpoint_path, normalizer_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the frozen, locally generated checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    class_labels = checkpoint["class_labels"]
    label_to_id = {
        label: index
        for index, label in enumerate(class_labels)
    }
    num_classes = len(class_labels)

    # Load the existing training-fitted normalizer.
    normalizer = load_normalizer(normalizer_path)

    # Load the held-out test manifest.
    manifest = pd.read_csv(manifest_path)

    feature_paths = []
    labels = []

    for _, row in manifest.iterrows():
        label = str(row["label"])

        if label not in label_to_id:
            raise ValueError(f"Unknown test label: {label}")

        feature_path = (
            FEATURE_DIR / "test" /
            (str(row["video_path"]) + ".npz")
        )

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        feature_paths.append(feature_path)
        labels.append(label_to_id[label])

    if len(feature_paths) != 190:
        raise ValueError(
            f"Expected 190 test samples, found {len(feature_paths)}"
        )

    base_dataset = SignSequenceDataset(feature_paths, labels)
    dataset = NormalizedDataset(base_dataset, normalizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sign_sequences,
    )

    # Restore the exact selected model.
    model = BiLSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []
    top5_correct = 0

    with torch.no_grad():
        for sequences, targets, lengths in loader:
            outputs = model(sequences.to(DEVICE), lengths)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(targets.numpy())
            all_predictions.extend(predictions.cpu().numpy())

            top5 = outputs.topk(5, dim=1).indices
            top5_correct += (
                top5 == targets.unsqueeze(1)
            ).any(dim=1).sum().item()

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )

    top5_accuracy = top5_correct / len(all_labels)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=np.arange(num_classes),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "test_classification_report.csv"
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=np.arange(num_classes),
    )

    pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    ).to_csv(
        OUTPUT_DIR / "test_confusion_matrix.csv"
    )

    summary = {
        "checkpoint_epoch": checkpoint["epoch"],
        "test_samples": len(all_labels),
        "num_classes": num_classes,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "top5_accuracy": float(top5_accuracy),
        "validation_macro_f1": float(
            checkpoint["validation_macro_f1"]
        ),
        "training_or_tuning_on_test": False,
    }

    import json

    with open(OUTPUT_DIR / "test_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("=" * 55)
    print("BiLSTM V1 - FINAL TEST EVALUATION")
    print("=" * 55)
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
