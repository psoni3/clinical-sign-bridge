from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from app.dataset.manifest import DatasetManifestError, load_manifest, manifest_sha256
from app.dataset.preprocessing import preprocess_manifest
from app.dataset.splits import assign_signer_separated_splits


def _write_report(path: Path, content: str, manifest_path: Path) -> None:
    if path.resolve() == manifest_path.resolve():
        raise ValueError("output path must not overwrite the source manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.dataset",
        description="Validate governed ISL dataset metadata and generate signer-separated splits.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a versioned dataset manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument(
        "--training-ready",
        action="store_true",
        help="require consent, governance, license and annotation approvals",
    )
    split = commands.add_parser("split", help="generate deterministic signer-separated splits")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", type=int, default=17)
    preprocess = commands.add_parser(
        "preprocess",
        help="extract consent-gated MediaPipe landmarks into a controlled output directory",
    )
    preprocess.add_argument("manifest", type=Path)
    preprocess.add_argument("--media-root", required=True, type=Path)
    preprocess.add_argument("--output-dir", required=True, type=Path)
    preprocess.add_argument("--target-fps", type=int, default=15)
    preprocess.add_argument("--max-frames", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == "validate":
            if arguments.training_ready:
                manifest.require_training_approval()
            print(
                f"Valid dataset manifest: version={manifest.dataset_version}, "
                f"samples={len(manifest.samples)}, sha256={manifest_sha256(manifest)}"
            )
            return 0

        if arguments.command == "split":
            report = assign_signer_separated_splits(manifest, arguments.seed)
            _write_report(arguments.output, report.to_json(), arguments.manifest)
            print(
                f"Created signer-separated split report: version={report.dataset_version}, "
                f"sample_counts={report.sample_counts}, signer_counts={report.signer_counts}"
            )
            return 0

        index_path = preprocess_manifest(
            manifest,
            arguments.media_root,
            arguments.output_dir,
            target_fps=arguments.target_fps,
            max_frames=arguments.max_frames,
        )
        print(f"Created landmark feature index outside source media: {index_path}")
        return 0
    except (DatasetManifestError, OSError, ValueError) as error:
        print(f"Dataset command failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
