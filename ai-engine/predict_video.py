import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clinicalsignbridge_ai.data.sequence_dataset import SignSequenceDataset  # noqa: E402, I001
from clinicalsignbridge_ai.data.sequence_normalizer import SequenceNormalizer  # noqa: E402
from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier  # noqa: E402
from extract_include_landmarks import process_video  # noqa: E402


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bilstm"
CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"
NORMALIZATION_PATH = ARTIFACT_DIR / "normalization.npz"
LABELS_PATH = ARTIFACT_DIR / "labels.json"
INPUT_SIZE = 228
TOP_K = 5
DEVICE = torch.device("cpu")


def load_label_mapping(path: Path, checkpoint_labels: list[str]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")

    with path.open(encoding="utf-8") as file:
        label_to_id = json.load(file)

    if not isinstance(label_to_id, dict):
        raise ValueError(f"Class mapping must be a JSON object: {path}")

    if any(
        not isinstance(label, str)
        or not isinstance(class_id, int)
        or isinstance(class_id, bool)
        for label, class_id in label_to_id.items()
    ):
        raise ValueError(f"Class mapping must contain string labels and integer IDs: {path}")

    if len(label_to_id) != len(checkpoint_labels) or len(label_to_id) < TOP_K:
        raise ValueError(
            f"Class mapping must contain at least {TOP_K} classes and match the checkpoint."
        )

    class_ids = sorted(label_to_id.values())
    if class_ids != list(range(len(label_to_id))):
        raise ValueError("Class mapping IDs must be unique and contiguous from zero.")

    labels_by_id = [
        label
        for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]
    if labels_by_id != checkpoint_labels:
        raise ValueError("Class mapping order does not match the checkpoint.")

    return labels_by_id


def load_normalizer(path: Path) -> SequenceNormalizer:
    if not path.is_file():
        raise FileNotFoundError(f"Training normalizer not found: {path}")

    normalizer = SequenceNormalizer()
    try:
        with np.load(path, allow_pickle=False) as data:
            for name, (start, end, _) in normalizer.GROUPS.items():
                mean = data[f"{name}_mean"].astype(np.float32, copy=True)
                std = data[f"{name}_std"].astype(np.float32, copy=True)
                expected_shape = (end - start,)
                if mean.shape != expected_shape or std.shape != expected_shape:
                    raise ValueError(
                        f"Incompatible {name} normalization dimensions in {path}."
                    )
                if not np.isfinite(mean).all() or not np.isfinite(std).all():
                    raise ValueError(f"Non-finite {name} normalization values in {path}.")
                if (std <= 0).any():
                    raise ValueError(f"Invalid {name} standard deviations in {path}.")
                normalizer.statistics[name] = {"mean": mean, "std": std}
    except KeyError as error:
        raise ValueError(
            f"Missing normalization statistic {error.args[0]!r} in {path}."
        ) from error

    normalizer.fitted = True
    return normalizer


def extract_sequence(video_path: Path) -> tuple[torch.Tensor, int]:
    with tempfile.TemporaryDirectory(prefix="clinical-sign-bridge-") as temp_dir:
        feature_path = Path(temp_dir) / "video.npz"
        extraction = process_video(
            video_path,
            feature_path,
            label="",
            relative_path=video_path.name,
        )
        sequence, _, frame_count = SignSequenceDataset([feature_path], [0])[0]

    if sequence.ndim != 2 or sequence.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"Expected extracted input shape [frames, {INPUT_SIZE}], "
            f"got {tuple(sequence.shape)}."
        )
    if frame_count == 0:
        raise ValueError("The video produced an empty landmark sequence.")
    if not torch.isfinite(sequence).all():
        raise ValueError("The video produced non-finite landmark features.")
    if frame_count != extraction["frames"]:
        raise ValueError("Extracted frame count does not match the sequence length.")

    return sequence, frame_count


def predict(video_path: Path) -> tuple[int, list[tuple[str, float]]]:
    for path in (CHECKPOINT_PATH, NORMALIZATION_PATH, LABELS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference artifact not found: {path}")

    checkpoint_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(checkpoint_globals):
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    required_fields = (
        "input_size",
        "hidden_size",
        "num_layers",
        "num_classes",
        "dropout",
        "class_labels",
        "model_state_dict",
    )
    if not isinstance(checkpoint, dict) or any(
        field not in checkpoint for field in required_fields
    ):
        raise ValueError(f"Incompatible BiLSTM checkpoint: {CHECKPOINT_PATH}")
    if checkpoint["input_size"] != INPUT_SIZE:
        raise ValueError(
            f"Checkpoint expects {checkpoint['input_size']} inputs per frame; "
            f"expected {INPUT_SIZE}."
        )
    if not isinstance(checkpoint["class_labels"], list) or not all(
        isinstance(label, str) for label in checkpoint["class_labels"]
    ):
        raise ValueError("Checkpoint class labels have an incompatible format.")

    labels = load_label_mapping(LABELS_PATH, checkpoint["class_labels"])
    num_classes = checkpoint["num_classes"]
    if num_classes != len(labels):
        raise ValueError("Checkpoint class count does not match the class mapping.")

    normalizer = load_normalizer(NORMALIZATION_PATH)
    sequence, frame_count = extract_sequence(video_path)
    sequence = normalizer.transform(sequence)
    if sequence.shape[1] != checkpoint["input_size"]:
        raise ValueError(
            f"Model input width mismatch: got {sequence.shape[1]}, "
            f"expected {checkpoint['input_size']}."
        )

    model = BiLSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=num_classes,
        dropout=checkpoint["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    lengths = torch.tensor([frame_count], dtype=torch.long)
    with torch.inference_mode():
        logits = model(sequence.unsqueeze(0).to(DEVICE), lengths)
        if logits.shape != (1, num_classes):
            raise ValueError(
                f"Model returned incompatible output shape {tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Model returned non-finite prediction scores.")
        scores = torch.softmax(logits[0], dim=0)
        top_scores, top_indices = torch.topk(scores, TOP_K)

    predictions = [
        (labels[index], float(score))
        for index, score in zip(
            top_indices.tolist(),
            top_scores.tolist(),
            strict=True,
        )
    ]
    return frame_count, predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify a sign-language video with the frozen BiLSTM V1 model."
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    args = parser.parse_args()

    video_path = args.video.expanduser()
    if not video_path.exists():
        parser.error(f"video does not exist: {video_path}")
    if not video_path.is_file():
        parser.error(f"video path is not a file: {video_path}")

    try:
        frame_count, predictions = predict(video_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")

    print(f"Input: {video_path.name}")
    print(f"Frames: {frame_count}")
    print("Top predictions:")
    for rank, (label, score) in enumerate(predictions, start=1):
        print(f"{rank}. {label}: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
