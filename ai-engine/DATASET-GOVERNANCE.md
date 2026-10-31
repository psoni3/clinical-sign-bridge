# ISL Dataset Governance and Annotation Draft

**Status: DRAFT — not approved for data collection, training, or redistribution.**

This draft is an implementation aid, not consent language or legal advice. No ISL
expert, Deaf signer, healthcare domain expert, privacy reviewer, or licensing authority
has approved it in this repository. Do not collect or import signer recordings until
the responsible human reviewers approve the purpose, participant materials, storage,
retention, withdrawal, access, and reuse terms.

## Required human approvals before collection or training

Record approval evidence outside source control and reference it only with an opaque
identifier in dataset metadata. The review group should include:

- qualified ISL linguists and Deaf/native or fluent ISL signers;
- healthcare workflow and clinical safety reviewers;
- privacy/legal reviewers familiar with applicable Indian requirements;
- security and data-stewardship owners; and
- licensing owners for every external dictionary, example, or reference resource.

The project team must document who can access recordings and derived landmarks,
where encryption and backups are controlled, how access is revoked, and how an
incident or withdrawal is handled. No contributor identity or contact detail belongs
in manifests, reports, model cards, or this repository.

## Separate purposes and participant choices

1. **Live processing** is the consultation workflow's separate purpose. Its existing
   consent does not authorize research collection or model training.
2. **Research/video collection** requires a separate, optional, explicit consent
   process with an approved policy version and a clear retention and withdrawal path.
3. **Model training and evaluation** require a separate, explicit purpose grant for
   each sample. Consent must explain that derived landmark sequences and trained
   parameters can persist and that removing an individual contribution from a trained
   artifact may require retraining.
4. Refusal or withdrawal must not affect clinical care or access to a qualified
   interpreter. Collection and training remain disabled until a human-approved
   withdrawal procedure is available.
5. On withdrawal, stop future use, quarantine/delete affected source and derived
   artifacts according to the approved policy, issue a new immutable dataset version,
   and assess whether dependent checkpoints must be withdrawn or retrained. Do not
   rewrite old manifests to conceal prior use.

The actual wording, legal basis, retention periods, storage region, access grants, and
withdrawal service levels are intentionally unresolved; competent reviewers must set
them before collection.

## License and permitted-use review

For each dictionary or external dataset, document the source, license identifier,
permitted research/training use, redistribution restrictions, attribution duties,
expiration/withdrawal terms, and the approving reviewer. Public availability is not
permission to train or redistribute. Keep the actual license evidence in the
controlled project record, not in this code repository.

## Draft annotation conventions

This guide is a starting point for ISL expert review, not a standardized or approved
gloss vocabulary.

- Use one uppercase `GLOSS` token per annotated temporal segment. Use `UNKNOWN` when
  annotators cannot reach a supported label; never infer a gloss from clinical context.
- Preserve the original sign sequence and boundaries. Annotate start/end offsets in
  milliseconds relative to the source clip; end must be greater than start and no
  later than the clip duration.
- Record regional/lexical variants separately. Do not collapse distinct valid signs
  into one label without documented expert adjudication.
- Capture negation, uncertainty, time, body location, intensity, and non-manual
  markers only with reviewer-approved labels; do not convert them into a sentence
  inferred from medical plausibility.
- Record quality flags (for example `OCCLUDED_HAND`, `LOW_LIGHT`, `OUT_OF_FRAME`,
  `AMBIGUOUS_BOUNDARY`) without recording participant names or clinical details.
- Require annotator pseudonyms and a separate reviewer pseudonym. Disagreements remain
  `DISPUTED` until resolved under the approved adjudication process.
- The current baseline task is isolated gloss classification only. It is not sentence
  translation, medical interpretation, diagnosis, or an approved vocabulary.

## Dataset manifest and reproducible splits

The versioned JSON manifest schema is implemented in `app/dataset/models.py`.
Every sample uses an opaque sample ID, pseudonymous signer and consent references,
purpose-specific active consent, media checksum/reference, gloss segments, quality
flags, dataset version, and annotation/reviewer status. Media references are
relative to a separately controlled media root. Raw recordings and derived feature
artifacts are ignored by Git; keep them in approved, access-controlled storage.

Validate a manifest without declaring it training-ready:

```powershell
uv run --project ai-engine python -m app.dataset validate path\to\manifest.json
```

Training-ready validation and split generation fail closed unless governance,
license, annotation-guide and per-sample review/consent fields are approved:

```powershell
uv run --project ai-engine python -m app.dataset validate path\to\manifest.json --training-ready
uv run --project ai-engine python -m app.dataset split path\to\manifest.json --output path\outside\repo\split-report.json --seed 17
```

Dataset versions are immutable. Any addition, removal, consent withdrawal, corrected
annotation, license change, or split change requires a new version and a new content
digest. Split assignments are by signer; sample-level random splitting is prohibited.

## Baseline and model approval

The temporal baseline is scoped to reviewed isolated gloss labels. Its training command
must reject a manifest without training-ready approvals. Training output starts in
`PENDING_REVIEW`; only designated independent reviewers may approve it after signer-
separated metrics, calibration, semantic error review, subgroup analysis, and safety
review are complete. A model card must bind the artifact checksum to the dataset
manifest and split-report digests.

The inference service remains on the deterministic abstaining mock by default. A
temporal model can load only from a configured model directory whose card is approved,
whose weights match the recorded checksum, and whose independent semantic review and
calibration are approved. Missing features, low confidence, or an explicit unknown
class abstain. A high score is not a clinical correctness guarantee.

No dataset, trained ISL artifact, measured score, or independent approval is included
in this repository. Model rollback means changing the configured artifact to a
previously approved immutable model card and verifying its checksum; the mock remains
the safe default if no approved artifact is available.
