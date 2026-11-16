
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for isolated sign classification.

    Input:
        landmarks: [batch, sequence_length, 228]
        lengths:   [batch]

    Per-frame features:
        75 landmarks * 3 coordinates = 225
        Pose, left-hand and right-hand detection masks = 3
        Total = 228
    """

    def __init__(
        self,
        input_size=228,
        hidden_size=128,
        num_layers=2,
        num_classes=50,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, landmarks, lengths):
        packed = pack_padded_sequence(
            landmarks,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # Final forward and backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        return self.classifier(combined)
