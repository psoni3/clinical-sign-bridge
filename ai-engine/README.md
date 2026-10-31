# AI inference and dataset tooling

The service defaults to `deterministic-mock`, which returns `UNSUPPORTED` with no
candidates. It does not recognize ISL. M4 adds governance-aware dataset manifests,
deterministic signer-separated splitting, MediaPipe landmark preprocessing, and an
optional temporal gloss-classification baseline. None of these changes includes
consented signer data or a trained/validated model.

## Optional dependency groups

```powershell
uv sync --project ai-engine --group dev
uv sync --project ai-engine --extra preprocessing
uv sync --project ai-engine --extra training
```

Keep dataset manifests, source media, extracted features, and model artifacts in
approved storage outside this repository. Only metadata with opaque references belongs
in source control. The `ai-engine` data/artifact directories and NumPy/PyTorch artifacts
are Git-ignored.

## Dataset lifecycle

Read [DATASET-GOVERNANCE.md](DATASET-GOVERNANCE.md) before using any dataset. That file
is explicitly a draft and is not approval to collect or train. Use one immutable
versioned JSON manifest per dataset release. Manifests must contain only opaque IDs and
must independently record model-training consent, annotation review, and license and
governance approval.

```powershell
uv run --project ai-engine python -m app.dataset validate path\to\manifest.json
uv run --project ai-engine python -m app.dataset validate path\to\manifest.json --training-ready
uv run --project ai-engine python -m app.dataset split path\to\manifest.json --output path\outside\repo\split-report.json --seed 17
uv run --project ai-engine python -m app.dataset preprocess path\to\manifest.json --media-root path\to\controlled-media --output-dir path\outside\repo\features
```

Split generation requires at least three signers and approved training metadata. It is
deterministic for a given manifest and seed, groups every sample by signer, and writes
the manifest digest and split counts into the report.

## Preprocessing and training

Install the optional groups shown above. Preprocessing accepts only manifests that pass
training-readiness validation; it reads clips from a configured media root and writes
derived landmark artifacts only to a caller-selected controlled output directory.
The model is an isolated gloss classifier, not a continuous ISL translation system.
Training artifacts are marked `PENDING_REVIEW` and cannot be loaded by the inference
provider until the model card and independent semantic/calibration reviews are approved.

```powershell
uv run --project ai-engine python -m app.training --help
uv run --project ai-engine python -m app.training path\to\manifest.json --split-report path\outside\repo\split-report.json --feature-index path\outside\repo\features\index.json --feature-dir path\outside\repo\features --output-dir path\outside\repo\model-artifacts\release-0.1.0 --seed 17 --epochs 50
```

Preprocessing verifies every clip SHA-256 against its manifest entry, resolves each
media reference under the configured media root, samples up to 96 frames, and writes
the 258-dimensional pose/hand landmark sequences and a checksummed index atomically.
The output directory must not already exist. Training verifies the manifest, split
report, feature checksums, and signer assignments before fitting the temporal
mean-and-motion-delta MLP. It emits validation/test accuracy, macro-F1, calibration
error, Brier score, per-gloss precision/recall/F1 and confusion matrices only as
research diagnostics; it does not assert an acceptance threshold or certify model
quality.

For normal local development and Compose, no optional ML dependency is installed and
the deterministic mock remains active. To opt into a model provider, set
`CLINICAL_INFERENCE_PROVIDER=temporal-baseline` and
`CLINICAL_MODEL_DIR` to an approved model directory. Startup fails explicitly for a
missing or unapproved artifact; it never silently falls back to a model or fabricates a
candidate.
An approved card must also name the independent semantic and calibration review
approvals, an opaque approval reference, and the reviewed minimum confidence threshold.
The training tool deliberately emits a `PENDING_REVIEW` card; no command auto-approves
it. The provider accepts at most 96 finite frames of 258 features each and abstains if
features are absent or confidence is below the approved threshold.

## Current status

- Dataset schema, validation, version digest, signer-level split report, preprocessing
  and baseline training/evaluation tooling are engineering scaffolding.
- Human expert engagement, approval of consent/governance/licensing/gloss conventions,
  actual data collection, training, independent evaluation and model approval remain
  prerequisites and are not represented as complete.
- The recognition API contract and provider are opt-in. The consultation backend
  continues to use its existing abstaining mock workflow.
