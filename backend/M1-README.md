# M1 consultation and conversation API

M1 adds backend consultation, temporary session, consent, conversation,
recognition, interpretation, confirmation, and notification workflows. It is
still a local prototype: use synthetic data only and do not expose it publicly.
No login, user accounts, or clinically validated ISL recognition are implemented.

The API is rooted at `/api/v1`. JSON request and response fields use camelCase;
the backend also accepts snake_case request fields for local compatibility. OpenAPI is
available at `http://localhost:8000/docs` when the Compose stack is running.

## Local API flow

1. `POST /consultations` creates a consultation and returns a one-time join code.
2. `POST /consultations/{id}/join` accepts the code and a `session_type` of
   `PATIENT_DEVICE` or `DOCTOR_VIEW`; the response includes temporary session
   identifiers. These are prototype handles, not production credentials.
3. `POST /consultations/{id}/start` starts it. `GET /consultations/{id}` returns
   its status and conversation ID.
4. The patient records `PROCESSING_FOR_LIVE_TRANSLATION` consent at
   `POST /consultations/{id}/consents`, including their `sessionId`.
   Posting `accepted: false` withdraws active consent for that purpose.
   Research consent uses the separate `RESEARCH_VIDEO_COLLECTION` purpose.
5. `POST /consultations/{id}/recognitions` accepts multipart fields `video`,
   `clientRequestId`, and `sessionId`. It accepts MP4 (`.mp4`,
   `video/mp4`, ISO-BMFF `ftyp`) and MOV (`.mov`, `video/quicktime`,
   QuickTime `ftyp` brand) uploads. WebM and other unsupported or mismatched
   media types receive HTTP 415; invalid container signatures receive HTTP
   400. The service bounds reads to 50 MiB, validates consent/session binding,
   and stores only upload metadata plus a SHA-256 digest, not the uploaded
   bytes.
6. Recognition claims are committed as `PROCESSING` before inference with a
   180-second lease by default (`CLINICAL_RECOGNITION_PROCESSING_LEASE_SECONDS`,
   valid from 1 to 3600 seconds). A repeated request key with identical video
   returns the existing completed or active recognition; changed content with
   that key conflicts. Expired claims and `FAILED` recognitions can be retried
   with the same key and content. Each attempt receives a fencing token, and
   only its current unexpired token can persist completion or failure. Legacy
   `PROCESSING` rows keep null lease fields and are immediately reclaimable.
   Drain pre-lease API workers before applying the migration so older in-flight
   attempts cannot complete without checking a fencing token.
   Successful inference atomically completes the recognition and stores an
   `UNVERIFIED` interpretation and ranked candidates.
   Candidate text remains the raw INCLUDE-50 class label, confidence is null,
   and the interpretation response warns that labels are not clinical phrases
   and scores are not confidence. These candidates do not create a confirmation
   or patient message; the patient must explicitly confirm one through the
   confirmation workflow. Inference failures produce `FAILED` without an
   interpretation or candidates.
   The current AI service accepts MOV/MP4; WebM uploads are rejected by the
   backend, so use MP4 or MOV for successful recognition.
7. Patient typed/visual messages and doctor-authored messages are persisted at
   `/conversations/{id}/messages`. History requires `session_id` and is paged
   with `offset` and `limit`. A patient message is marked `PATIENT_CONFIRMED`
   only when the patient explicitly submits it; the server sets its status and
   confirmation timestamp. For `DOCTOR_VIEW` sessions, unconfirmed patient
   messages are excluded in the database query before the count and page are
   computed. Doctor-authored messages remain in the timeline. Acknowledgement
   is limited to confirmed patient messages and is tracked separately.
8. Confirm/correct/reject endpoints validate the interpretation, patient session,
   consultation, and candidate, then persist confirmation, message, state, and
   audit and outbox data in one database transaction. Confirmation and correction
   first flush a confirmation with a null message reference, then flush the message
   referencing that confirmation, and finally link the confirmation back to the
   message before the single commit. Both immediate foreign keys remain enforced.
   Database failures log the correlation ID, exception type, and traceback with
   exception messages/SQL parameters redacted.
   Confirmation and correction
   claim the `UNVERIFIED`/`ACCEPTABLE` interpretation with a conditional database
   update; concurrent requests cannot both create a confirmed message. Replaying the
   same confirmation or correction idempotency key and payload returns the
   original result; a changed payload conflicts. Reject and clarification
   transitions also require an `UNVERIFIED` interpretation and create no
   patient message. Re-signing starts a new recognition rather than confirming
   the prior interpretation. The patient may explicitly confirm a persisted
   candidate, but candidate text remains an unverified dataset label.
9. `GET /doctors/consultations` lists active prototype consultations.

All interpretation/recognition reads and message-history requests require the
associated `sessionId` query parameter.
Errors use a standard `code`, `message`, `correlationId`, and optional
`fieldErrors` response body; the same correlation ID is returned in
`X-Correlation-ID`.

## WebSocket notifications

Connect to `/api/v1/ws/consultations/{consultationId}?sessionId={sessionId}`.
The server validates the temporary session. Notifications are stored in the
transactional outbox and delivered after commit, including consultation,
recognition, confirmation, and message events. REST remains authoritative; fetch
REST history after connecting or reconnecting.

Connections and broadcasts are in-memory and process-local. Outbox delivery is
retried, but WebSocket events are not replayed after reconnect or across multiple
API workers.

## Limitations

- M1 has no authentication or production authorization; only synthetic local
  data is appropriate.
- The BiLSTM inference is a research prototype, not clinically validated.
  Duration is recorded as unknown, so a 30-second limit cannot be verified yet;
  the 50 MiB bound and container signature check are not a substitute for full
  media validation.
- The join-attempt limiter is process-local and cannot protect a multi-worker or
  hosted deployment.
- The doctor consultation list is unauthenticated by design in this prototype.
- This backend integration does not change the existing Angular video
  prediction workflow or add a new browser-facing inference endpoint.

Run backend tests and checks from the repository root as described in the root
README. Start the entire local stack with `docker compose --env-file .env.example
up --build` or use a local `.env` file.

### Confirmation foreign-key regression tests

The regression tests always run with SQLite foreign-key enforcement. To also run
against PostgreSQL, supply a **dedicated disposable test database**, never the
consultation database. Apply the existing migrations to that database first.
The tests add synthetic rows and create missing tables, but do not reset or delete
database contents. Run from the repository root in PowerShell:

```powershell
# Set this to the connection URL of your dedicated PostgreSQL test database.
$env:CLINICAL_TEST_POSTGRES_URL = '<dedicated-postgresql-test-url>'
uv run --project backend pytest backend\tests\test_confirmation_foreign_keys.py -q
```

PostgreSQL cases are explicitly skipped when `CLINICAL_TEST_POSTGRES_URL` is unset;
SQLite results alone are not PostgreSQL validation.
