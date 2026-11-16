
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_sign_sequences(batch):
    """
    Pads variable-length sign sequences within a batch.

    Each dataset item contains:
        sequence: [T, 228]
        label: scalar
        length: integer
    """

    sequences, labels, lengths = zip(*batch)

    sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sequences, labels, lengths
