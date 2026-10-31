import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from app.dataset.models import DatasetManifest


class DatasetManifestError(ValueError):
    """Raised when dataset metadata cannot be safely loaded or validated."""


def load_manifest(path: Path) -> DatasetManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetManifestError(
            f"Invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error
    except OSError as error:
        raise DatasetManifestError("Could not read dataset manifest") from error

    try:
        return DatasetManifest.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
        raise DatasetManifestError(f"Dataset manifest validation failed: {details}") from error


def manifest_sha256(manifest: DatasetManifest) -> str:
    payload = manifest.model_dump(mode="json", by_alias=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
