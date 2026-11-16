
import json
import random
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

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from clinicalsignbridge_ai.data.sequence_dataset import SignSequenceDataset
from clinicalsignbridge_ai.data.sequence_collator import (
    collate_sign_sequences,
)
from clinicalsignbridge_ai.data.sequence_normalizer import SequenceNormalizer
from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier


DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests" / "frozen"
FEATURE_DIR = PROJECT_ROOT / "artifacts" / "features" / "INCLUDE"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "bilstm"

TRAIN_MANIFEST = MANIFEST_DIR / "train.csv"
VALIDATION_MANIFEST = MANIFEST_DIR / "validation.csv"

# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 100
PATIENCE = 15

INPUT_SIZE = 228
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

class NormalizedDataset(Dataset):
    """Applies training-fitted normalization to each sequence."""

    def __init__(self, base_dataset, normalizer):
        self.base_dataset = base_dataset
        self.normalizer = normalizer

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sequence, label, length = self.base_dataset[index]

        sequence = self.normalizer.transform(sequence)

        return sequence, label, length


def load_manifest(split):
    manifest_path = MANIFEST_DIR / f"{split}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    return pd.read_csv(manifest_path)


def build_dataset(manifest, label_to_id, split):
    feature_paths = []
    labels = []

    for _, row in manifest.iterrows():
        label = str(row["label"])
        video_path = str(row["video_path"])

        if label not in label_to_id:
            raise ValueError(
                f"Unknown label '{label}' in {split} manifest."
            )

        feature_path = (
            FEATURE_DIR / split / (video_path + ".npz")
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        feature_paths.append(feature_path)
        labels.append(label_to_id[label])

    return SignSequenceDataset(feature_paths, labels)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, loader, criterion):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0
    top5_correct = 0

    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            top5 = outputs.topk(5, dim=1).indices
            top5_correct += (
                top5 == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    top5_accuracy = top5_correct / total_samples

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top5_accuracy": top5_accuracy,
        "labels": np.array(all_labels),
        "predictions": np.array(all_predictions),
    }


# ---------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------

def main():
    global NUM_CLASSES

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BiLSTM Training - INCLUDE-50")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Project: {PROJECT_ROOT}")

    # Load only train and validation manifests.
    train_manifest = load_manifest("train")
    validation_manifest = load_manifest("validation")

    # Create a stable class mapping from training labels.
    class_labels = sorted(
        train_manifest["label"].astype(str).unique().tolist()
    )

    label_to_id = {
        label: index
        for index, label in enumerate(class_labels)
    }

    NUM_CLASSES = len(class_labels)

    if NUM_CLASSES != 50:
        raise ValueError(
            f"Expected 50 classes, found {NUM_CLASSES}."
        )

    validation_labels = set(
        validation_manifest["label"].astype(str).unique()
    )

    unknown_labels = validation_labels - set(class_labels)

    if unknown_labels:
        raise ValueError(
            f"Validation contains unknown labels: {unknown_labels}"
        )

    print(f"Classes: {NUM_CLASSES}")
    print(f"Training samples: {len(train_manifest)}")
    print(f"Validation samples: {len(validation_manifest)}")

    # Build datasets.
    train_base = build_dataset(
        train_manifest,
        label_to_id,
        "train",
    )

    validation_base = build_dataset(
        validation_manifest,
        label_to_id,
        "validation",
    )

    # Fit normalization using training data only.
    print("\nFitting sequence normalizer on training data...")

    normalizer = SequenceNormalizer()
    normalizer.fit(train_base)

    normalizer.save(
        OUTPUT_DIR / "normalization.npz"
    )

    train_dataset = NormalizedDataset(
        train_base,
        normalizer,
    )

    validation_dataset = NormalizedDataset(
        validation_base,
        normalizer,
    )

    # Data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sign_sequences,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sign_sequences,
    )

    # Model.
    model = BiLSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = OUTPUT_DIR / "best_model.pt"

    print("\nStarting training...")
    print("-" * 60)

    # Training loop.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
        )

        val_macro_f1 = validation_metrics["macro_f1"]

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_metrics['loss']:.4f} | "
            f"Val Accuracy: {validation_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Val Top-5: {validation_metrics['top5_accuracy']:.4f}"
        )

        # Save the best model based on validation macro F1.
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "dropout": DROPOUT,
                    "class_labels": class_labels,
                    "validation_macro_f1": best_macro_f1,
                },
                best_model_path,
            )

            print("  -> Best model saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after {epoch} epochs. "
                f"No improvement for {PATIENCE} epochs."
            )
            break

    # -----------------------------------------------------
    # Final validation report using the best checkpoint
    # -----------------------------------------------------

    print("\nLoading best checkpoint for final validation report...")

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(
        model,
        validation_loader,
        criterion,
    )

    report = classification_report(
        final_metrics["labels"],
        final_metrics["predictions"],
        labels=np.arange(NUM_CLASSES),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "validation_classification_report.csv"
    )

    matrix = confusion_matrix(
        final_metrics["labels"],
        final_metrics["predictions"],
        labels=np.arange(NUM_CLASSES),
    )

    pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    ).to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as file:
        json.dump(
            label_to_id,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(
            final_metrics["macro_f1"]
        ),
        "validation_accuracy": float(
            final_metrics["accuracy"]
        ),
        "validation_top5_accuracy": float(
            final_metrics["top5_accuracy"]
        ),
        "training_samples": len(train_manifest),
        "validation_samples": len(validation_manifest),
        "num_classes": NUM_CLASSES,
        "test_set_evaluated": False,
    }

    with open(OUTPUT_DIR / "training_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Validation top-5 accuracy: {final_metrics['top5_accuracy']:.4f}")

    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print("Test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
