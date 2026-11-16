from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
import torch

import predict_video
from app.inference.video_predictions import (
    InvalidVideoError,
    VideoFrameLimitExceededError,
)
from clinicalsignbridge_ai.data.sequence_dataset import SignSequenceDataset
from clinicalsignbridge_ai.models.bilstm import BiLSTMClassifier
from extract_include_landmarks import (
    VideoFrameLimitExceededError as ExtractorFrameLimitExceededError,
)
from extract_include_landmarks import (
    VideoReadError,
    process_video,
)


class BilstmV1Service:
    def __init__(self, artifact_dir: Path, max_video_frames: int) -> None:
        if max_video_frames < 1:
            raise ValueError("max_video_frames must be positive.")

        artifact_dir = artifact_dir.resolve()
        checkpoint_path = artifact_dir / "best_model.pt"
        normalizer_path = artifact_dir / "normalization.npz"
        labels_path = artifact_dir / "labels.json"
        for path in (checkpoint_path, normalizer_path, labels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required BiLSTM V1 artifact not found: {path}")

        checkpoint_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(checkpoint_globals):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
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
            raise ValueError("BiLSTM V1 checkpoint has an incompatible format.")
        if checkpoint["input_size"] != predict_video.INPUT_SIZE:
            raise ValueError("BiLSTM V1 checkpoint has an incompatible input size.")
        if not isinstance(checkpoint["class_labels"], list) or not all(
            isinstance(label, str) for label in checkpoint["class_labels"]
        ):
            raise ValueError("BiLSTM V1 checkpoint labels have an incompatible format.")

        self.labels = predict_video.load_label_mapping(
            labels_path,
            checkpoint["class_labels"],
        )
        num_classes = checkpoint["num_classes"]
        if num_classes != len(self.labels):
            raise ValueError("BiLSTM V1 checkpoint class count does not match labels.")

        self.normalizer = predict_video.load_normalizer(normalizer_path)
        self.model = BiLSTMClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            num_classes=num_classes,
            dropout=checkpoint["dropout"],
        ).to(torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.max_video_frames = max_video_frames
        self._inference_lock = Lock()

    def predict_video(self, video_path: Path) -> tuple[int, list[tuple[str, float]]]:
        with self._inference_lock:
            return self._predict_video_locked(video_path)

    def _predict_video_locked(
        self,
        video_path: Path,
    ) -> tuple[int, list[tuple[str, float]]]:
        sequence, frame_count = self._extract_sequence(video_path)
        sequence = self.normalizer.transform(sequence)
        if sequence.shape[1] != predict_video.INPUT_SIZE:
            raise RuntimeError("Extracted sequence has an incompatible model input size.")

        lengths = torch.tensor([frame_count], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(sequence.unsqueeze(0), lengths)
            if logits.shape != (1, len(self.labels)):
                raise RuntimeError("BiLSTM V1 returned an incompatible output shape.")
            if not torch.isfinite(logits).all():
                raise RuntimeError("BiLSTM V1 returned non-finite scores.")
            scores = torch.softmax(logits[0], dim=0)
            top_scores, top_indices = torch.topk(scores, predict_video.TOP_K)

        candidates = [
            (self.labels[index], float(score))
            for index, score in zip(
                top_indices.tolist(),
                top_scores.tolist(),
                strict=True,
            )
        ]
        return frame_count, candidates

    def _extract_sequence(self, video_path: Path) -> tuple[torch.Tensor, int]:
        try:
            with TemporaryDirectory(prefix="clinical-sign-bridge-features-") as temp_dir:
                feature_path = Path(temp_dir) / "video.npz"
                extraction = process_video(
                    video_path,
                    feature_path,
                    label="",
                    relative_path=video_path.name,
                    max_frames=self.max_video_frames,
                )
                sequence, _, frame_count = SignSequenceDataset([feature_path], [0])[0]
        except ExtractorFrameLimitExceededError as error:
            raise VideoFrameLimitExceededError from error
        except VideoReadError as error:
            raise InvalidVideoError from error
        except ValueError as error:
            raise InvalidVideoError from error

        if sequence.ndim != 2 or sequence.shape[1] != predict_video.INPUT_SIZE:
            raise InvalidVideoError
        if frame_count == 0 or frame_count != extraction["frames"]:
            raise InvalidVideoError
        if not torch.isfinite(sequence).all():
            raise InvalidVideoError

        return sequence, frame_count
