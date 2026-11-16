
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
ROOT = Path(".")
FEATURE_DIR = ROOT / "artifacts/features/INCLUDE"
MANIFEST_DIR = ROOT / "data/manifests/frozen"
OUTPUT_DIR = ROOT / "artifacts/baseline"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        landmarks = data["landmarks"].astype(np.float32)

        masks = {
            "pose": data["pose_detected"].astype(bool),
            "left": data["left_hand_detected"].astype(bool),
            "right": data["right_hand_detected"].astype(bool),
        }

    # Landmark groups:
    # Pose: 33 points, left hand: 21, right hand: 21
    groups = {
        "pose": landmarks[:, 0:33, :],
        "left": landmarks[:, 33:54, :],
        "right": landmarks[:, 54:75, :],
    }

    features = []

    for name, values in groups.items():
        valid = masks[name]

        # Preserve missingness explicitly.
        features.append(np.array([valid.mean()], dtype=np.float32))

        if valid.any():
            selected = values[valid]

            features.extend([
                selected.mean(axis=(0, 1)).astype(np.float32),
                selected.std(axis=(0, 1)).astype(np.float32),
            ])
        else:
            features.extend([
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ])

    return np.concatenate(features)


def load_split(split, label_to_id):
    manifest = pd.read_csv(MANIFEST_DIR / f"{split}.csv")

    X, y = [], []

    for _, row in manifest.iterrows():
        video_path = Path(str(row["video_path"]))
        feature_path = (
            FEATURE_DIR / split / (str(video_path) + ".npz")
        )

        X.append(extract_features(feature_path))
        y.append(label_to_id[str(row["label"])])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


# --------------------------------------------------
# Model
# --------------------------------------------------

class BaselineMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Dataset and labels
# --------------------------------------------------

train_manifest = pd.read_csv(MANIFEST_DIR / "train.csv")
labels = sorted(train_manifest["label"].astype(str).unique())

label_to_id = {label: index for index, label in enumerate(labels)}

X_train, y_train = load_split("train", label_to_id)
X_val, y_val = load_split("validation", label_to_id)

print(
    "Validation classes represented:",
    len(np.unique(y_val)),
    "out of",
    len(labels),
)

assert len(labels) == 50, f"Expected 50 classes, got {len(labels)}"

# Standardization fitted only on training data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

X_val_tensor = torch.tensor(X_val).to(DEVICE)
y_val_tensor = torch.tensor(y_val).to(DEVICE)

# --------------------------------------------------
# Training
# --------------------------------------------------

model = BaselineMLP(X_train.shape[1], len(labels)).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss()

best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0

    for features, targets in train_loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_predictions = val_logits.argmax(dim=1).cpu().numpy()

    validation_classes = np.unique(y_val)

    val_f1 = f1_score(
        y_val,
        val_predictions,
        labels=validation_classes,
        average="macro",
        zero_division=0,
    )

    train_loss = total_loss / len(train_dataset)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation Macro F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            OUTPUT_DIR / "best_model.pt",
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

# --------------------------------------------------
# Save model metadata and validation results
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():
    logits = model(X_val_tensor)
    probabilities = torch.softmax(logits, dim=1).numpy()

predictions = probabilities.argmax(axis=1)

top5 = np.argsort(probabilities, axis=1)[:, -5:]
top5_accuracy = np.mean([
    actual in predicted
    for actual, predicted in zip(y_val, top5)
])

print("\nVALIDATION RESULTS")
print("=" * 60)
print("Classes:", len(labels))
print("Training samples:", len(y_train))
print("Validation samples:", len(y_val))
print("Best epoch:", best_epoch)
print("Macro F1:", round(best_f1, 4))
print("Top-1 accuracy:", round(accuracy_score(y_val, predictions), 4))
print("Top-5 accuracy:", round(float(top5_accuracy), 4))

report = classification_report(
    y_val,
    predictions,
    labels=np.arange(len(labels)),
    target_names=labels,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "validation_classification_report.csv"
)

pd.DataFrame(
    confusion_matrix(
        y_val,
        predictions,
        labels=np.arange(len(labels)),
    ),
    index=labels,
    columns=labels,
).to_csv(OUTPUT_DIR / "validation_confusion_matrix.csv")

np.savez(
    OUTPUT_DIR / "preprocessing.npz",
    mean=scaler.mean_,
    scale=scaler.scale_,
    labels=np.asarray(labels),
)

print("\nSaved artifacts to:", OUTPUT_DIR.resolve())
