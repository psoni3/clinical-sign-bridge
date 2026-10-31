from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.dataset.manifest import manifest_sha256
from app.dataset.models import DatasetManifest

SplitName = Literal["train", "validation", "test"]
SPLIT_TARGETS: tuple[tuple[SplitName, float], ...] = (
    ("train", 0.70),
    ("validation", 0.15),
    ("test", 0.15),
)


@dataclass(frozen=True)
class DatasetSplit:
    dataset_version: str
    manifest_sha256: str
    seed: int
    sample_assignments: dict[UUID, SplitName]
    signer_assignments: dict[str, SplitName]
    sample_counts: dict[SplitName, int]
    signer_counts: dict[SplitName, int]

    def to_json(self) -> str:
        report = {
            "schemaVersion": "1.0",
            "datasetVersion": self.dataset_version,
            "manifestSha256": self.manifest_sha256,
            "seed": self.seed,
            "sampleAssignments": {
                str(sample_id): split
                for sample_id, split in sorted(self.sample_assignments.items())
            },
            "signerAssignments": dict(sorted(self.signer_assignments.items())),
            "sampleCounts": self.sample_counts,
            "signerCounts": self.signer_counts,
        }
        return json.dumps(report, sort_keys=True, indent=2) + "\n"


def assign_signer_separated_splits(
    manifest: DatasetManifest,
    seed: int,
) -> DatasetSplit:
    manifest.require_training_approval()
    signer_sample_ids: dict[str, list[UUID]] = defaultdict(list)
    for sample in manifest.samples:
        signer_sample_ids[sample.signer_pseudonym].append(sample.sample_id)
    if len(signer_sample_ids) < len(SPLIT_TARGETS):
        raise ValueError("at least three distinct signers are required for signer-separated splits")

    signer_order = sorted(
        signer_sample_ids,
        key=lambda signer_id: (
            -len(signer_sample_ids[signer_id]),
            hashlib.sha256(f"{seed}:{signer_id}".encode()).hexdigest(),
        ),
    )
    assigned_signers: dict[str, SplitName] = {}
    sample_loads: Counter[SplitName] = Counter()

    for index, signer_id in enumerate(signer_order):
        if index < len(SPLIT_TARGETS):
            split_name = SPLIT_TARGETS[index][0]
        else:
            split_name = min(
                SPLIT_TARGETS,
                key=lambda split: (
                    sample_loads[split[0]] / split[1],
                    split[0],
                ),
            )[0]
        assigned_signers[signer_id] = split_name
        sample_loads[split_name] += len(signer_sample_ids[signer_id])

    sample_assignments = {
        sample_id: assigned_signers[signer_id]
        for signer_id, sample_ids in signer_sample_ids.items()
        for sample_id in sample_ids
    }
    sample_counts = Counter(sample_assignments.values())
    signer_counts = Counter(assigned_signers.values())
    return DatasetSplit(
        dataset_version=manifest.dataset_version,
        manifest_sha256=manifest_sha256(manifest),
        seed=seed,
        sample_assignments=sample_assignments,
        signer_assignments=assigned_signers,
        sample_counts={split: sample_counts[split] for split, _ in SPLIT_TARGETS},
        signer_counts={split: signer_counts[split] for split, _ in SPLIT_TARGETS},
    )
