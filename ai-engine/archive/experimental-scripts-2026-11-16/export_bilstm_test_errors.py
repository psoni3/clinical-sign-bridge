
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinicalsignbridge_ai.data.sequence_dataset import SignSequenceDataset
from clinicalsignbridge_ai.data.sequence_collator import collate_sign_sequences
from clinicalsignbridge_ai.data.sequence_normalizer import SequenceNormalizer
from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier


FEATURE_DIR = PROJECT_ROOT / "artifacts/features/INCLUDE"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/frozen/test.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts/bilstm"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"
NORMALIZER_PATH = MODEL_DIR / "normalization.npz"
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_labels = checkpoint["class_labels"]
    label_to_id = {
        label: index for index, label in enumerate(class_labels)
    }

    normalizer = load_normalizer(NORMALIZER_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    feature_paths = []
    actual_ids = []

    for _, row in manifest.iterrows():
        label = str(row["label"])

        if label not in label_to_id:
            raise ValueError(f"Unknown label: {label}")

        feature_path = (
            FEATURE_DIR / "test" /
            (str(row["video_path"]) + ".npz")
        )

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        feature_paths.append(feature_path)
        actual_ids.append(label_to_id[label])

    base_dataset = SignSequenceDataset(feature_paths, actual_ids)
    dataset = NormalizedDataset(base_dataset, normalizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sign_sequences,
    )

    model = BiLSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    sample_index = 0

    with torch.no_grad():
        for sequences, targets, lengths in loader:
            logits = model(sequences.to(DEVICE), lengths)
            probabilities = torch.softmax(logits, dim=1)

            top_probs, top_ids = probabilities.topk(5, dim=1)

            for batch_index in range(len(targets)):
                actual_id = int(targets[batch_index])
                predicted_id = int(top_ids[batch_index, 0])

                if actual_id != predicted_id:
                    row = manifest.iloc[sample_index]

                    top5 = [
                        f"{class_labels[int(class_id)]} ({float(prob):.4f})"
                        for class_id, prob in zip(
                            top_ids[batch_index],
                            top_probs[batch_index],
                        )
                    ]

                    results.append({
                        "video_path": str(row["video_path"]),
                        "actual_class": class_labels[actual_id],
                        "predicted_class": class_labels[predicted_id],
                        "predicted_confidence": float(
                            top_probs[batch_index, 0]
                        ),
                        "actual_class_confidence": float(
                            probabilities[batch_index, actual_id]
                        ),
                        "top_5_predictions": " | ".join(top5),
                    })

                sample_index += 1

    errors = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "misclassified_samples.csv"
    errors.to_csv(output_path, index=False)

    print("=" * 55)
    print("BiLSTM V1 - TEST ERROR EXPORT")
    print("=" * 55)
    print(f"Test samples: {len(manifest)}")
    print(f"Correct predictions: {len(manifest) - len(errors)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Error report: {output_path}")

    if not errors.empty:
        print("\nMisclassification counts:")
        print(
            errors.groupby(
                ["actual_class", "predicted_class"]
            ).size().sort_values(ascending=False)
        )


if __name__ == "__main__":
    main()
