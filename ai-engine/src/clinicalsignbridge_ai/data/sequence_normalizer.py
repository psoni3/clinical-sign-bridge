
import numpy as np
import torch


class SequenceNormalizer:
    """
    Normalizes landmark coordinates using training data only.

    Pose, left-hand and right-hand coordinates are normalized
    independently. Detection masks are never normalized.
    """

    GROUPS = {
        "pose": (0, 99, 225),
        "left": (99, 162, 226),
        "right": (162, 225, 227),
    }

    def __init__(self):
        self.statistics = {}
        self.fitted = False

    def fit(self, dataset):
        collected = {
            name: [] for name in self.GROUPS
        }

        for index in range(len(dataset)):
            sequence, _, _ = dataset[index]
            sequence = sequence.numpy()

            for name, (start, end, mask_index) in self.GROUPS.items():
                valid_frames = sequence[:, mask_index] > 0.5

                if valid_frames.any():
                    collected[name].append(
                        sequence[valid_frames, start:end]
                    )

        for name, arrays in collected.items():
            if not arrays:
                raise ValueError(
                    f"No valid training observations for {name}"
                )

            values = np.concatenate(arrays, axis=0)

            mean = values.mean(axis=0).astype(np.float32)
            std = values.std(axis=0).astype(np.float32)

            # Avoid division by zero for constant coordinates.
            std[std < 1e-6] = 1.0

            self.statistics[name] = {
                "mean": mean,
                "std": std,
            }

        self.fitted = True
        return self

    def transform(self, sequence):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted first.")

        result = sequence.clone()

        for name, (start, end, mask_index) in self.GROUPS.items():
            valid_frames = result[:, mask_index] > 0.5

            mean = torch.as_tensor(
                self.statistics[name]["mean"],
                dtype=result.dtype,
                device=result.device,
            )

            std = torch.as_tensor(
                self.statistics[name]["std"],
                dtype=result.dtype,
                device=result.device,
            )

            result[:, start:end] = (
                result[:, start:end] - mean
            ) / std

            # Keep missing detections as zero after normalization.
            result[~valid_frames, start:end] = 0.0

        return result

    def save(self, path):
        if not self.fitted:
            raise RuntimeError("Normalizer has not been fitted.")

        np.savez(
            path,
            **{
                f"{name}_{key}": value
                for name, stats in self.statistics.items()
                for key, value in stats.items()
            },
        )
