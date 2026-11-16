
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
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "bilstm"

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


def main():
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    normalizer_path = OUTPUT_DIR / "normalization.npz"

    if not checkpoint_path.exists() or not normalizer_path.exists():
        raise FileNotFoundError("BiLSTM checkpoint or normalizer is missing.")

    # This is a locally generated checkpoint from this project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    class_labels = checkpoint["class_labels"]
    label_to_id = {label: i for i, label in enumerate(class_labels)}
    num_classes = len(class_labels)

    # Restore the normalizer fitted on training data.
    normalizer = SequenceNormalizer()

    with np.load(normalizer_path, allow_pickle=False) as data:
        for name in normalizer.GROUPS:
            normalizer.statistics[name] = {
                "mean": data[f"{name}_mean"].copy(),
                "std": data[f"{name}_std"].copy(),
            }

    normalizer.fitted = True

    # Load validation only.
    manifest = pd.read_csv(MANIFEST_DIR / "validation.csv")

    feature_paths = []
    labels = []

    for _, row in manifest.iterrows():
        label = str(row["label"])

        if label not in label_to_id:
            raise ValueError(f"Unknown validation label: {label}")

        feature_path = (
            FEATURE_DIR / "validation" /
            (str(row["video_path"]) + ".npz")
        )

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        feature_paths.append(feature_path)
        labels.append(label_to_id[label])

    base_dataset = SignSequenceDataset(feature_paths, labels)
    dataset = NormalizedDataset(base_dataset, normalizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sign_sequences,
    )

    # Reconstruct the trained model.
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
        OUTPUT_DIR / "validation_classification_report.csv"
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
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    print("=" * 55)
    print("BiLSTM VALIDATION REPORT")
    print("=" * 55)
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")
    print("\nReports saved to:", OUTPUT_DIR)
    print("Test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
