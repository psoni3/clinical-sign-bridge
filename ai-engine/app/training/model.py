from __future__ import annotations

import importlib
from typing import Any


def build_classifier(class_count: int) -> Any:
    if class_count < 2:
        raise ValueError("the temporal baseline requires at least two gloss classes")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("install the AI package training extra to train a baseline") from error

    return torch.nn.Sequential(
        torch.nn.Linear(516, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(128, class_count),
    )


def temporal_descriptor(sequence: Any) -> Any:
    if sequence.ndim != 3 or sequence.shape[1] < 1 or sequence.shape[2] != 258:
        raise ValueError("features must have shape (batch, frames, 258)")
    temporal_mean = sequence.mean(dim=1)
    temporal_delta = sequence[:, -1, :] - sequence[:, 0, :]
    return importlib.import_module("torch").cat((temporal_mean, temporal_delta), dim=1)
