
import numpy as np
import torch
from torch.utils.data import Dataset


class SignSequenceDataset(Dataset):
    """
    Loads variable-length Holistic landmark sequences.

    Per-frame feature layout:
        0:225   Landmark coordinates (75 * 3)
        225     Pose detection mask
        226     Left-hand detection mask
        227     Right-hand detection mask
    """

    def __init__(self, feature_paths, labels):
        if len(feature_paths) != len(labels):
            raise ValueError("Feature paths and labels must have equal length.")

        self.feature_paths = list(feature_paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        path = self.feature_paths[index]
        label = int(self.labels[index])

        with np.load(path, allow_pickle=False) as data:
            landmarks = data["landmarks"].astype(np.float32)
            pose_mask = data["pose_detected"].astype(bool)
            left_mask = data["left_hand_detected"].astype(bool)
            right_mask = data["right_hand_detected"].astype(bool)

        frame_count = landmarks.shape[0]

        if landmarks.shape[1:] != (75, 3):
            raise ValueError(
                f"Expected landmarks shape [T, 75, 3], got {landmarks.shape}: {path}"
            )

        if not (
            len(pose_mask) == frame_count
            and len(left_mask) == frame_count
            and len(right_mask) == frame_count
        ):
            raise ValueError(f"Detection mask length mismatch: {path}")

        if frame_count == 0:
            raise ValueError(f"Empty sequence: {path}")

        # Copy coordinates so the original feature file remains immutable.
        coordinates = landmarks.copy()

        # Missing detections are represented by zero coordinates.
        # The corresponding masks preserve the distinction between
        # a missing detection and an actual coordinate observation.
        coordinates[~pose_mask, 0:33, :] = 0.0
        coordinates[~left_mask, 33:54, :] = 0.0
        coordinates[~right_mask, 54:75, :] = 0.0

        detection_masks = np.stack(
            [pose_mask, left_mask, right_mask],
            axis=1,
        ).astype(np.float32)

        sequence = np.concatenate(
            [
                coordinates.reshape(frame_count, 225),
                detection_masks,
            ],
            axis=1,
        )

        if not np.isfinite(sequence).all():
            raise ValueError(f"Non-finite values found: {path}")

        return (
            torch.from_numpy(sequence),
            torch.tensor(label, dtype=torch.long),
            frame_count,
        )
