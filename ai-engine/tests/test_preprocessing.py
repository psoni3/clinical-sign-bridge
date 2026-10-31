import hashlib

import pytest

from app.dataset.models import DatasetManifest, DatasetSample
from app.dataset.preprocessing import PreprocessingError, _resolve_media_file, preprocess_manifest


def test_preprocessing_rejects_media_checksum_mismatch(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    media_root = tmp_path / "media"
    clip_path = media_root / "clips" / "1.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"not-a-real-video")
    sample = approved_dataset_manifest.samples[0]

    with pytest.raises(PreprocessingError, match="checksum mismatch"):
        _resolve_media_file(media_root, sample)


def test_preprocessing_requires_training_approval_before_creating_output(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    manifest_data = approved_dataset_manifest.model_dump(mode="json")
    manifest_data["governance_status"] = "PENDING"
    manifest = DatasetManifest.model_validate(manifest_data)
    output_dir = tmp_path / "features"

    with pytest.raises(ValueError, match="governance approval"):
        preprocess_manifest(manifest, tmp_path / "missing-media", output_dir)

    assert not output_dir.exists()


def test_preprocessing_resolves_media_inside_configured_root(
    tmp_path,
    approved_dataset_manifest,
) -> None:
    media_root = tmp_path / "media"
    clip_path = media_root / "clips" / "1.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"controlled-test-clip")
    digest = hashlib.sha256(clip_path.read_bytes()).hexdigest()

    sample_data = approved_dataset_manifest.samples[0].model_dump(mode="json")
    sample_data["media_sha256"] = digest
    sample = DatasetSample.model_validate(sample_data)

    assert _resolve_media_file(media_root, sample) == clip_path.resolve()
