# M1 consultation and conversation API

M1 adds backend consultation, temporary session, consent, conversation,
recognition, interpretation, confirmation, and notification workflows. It is
still a local prototype: use synthetic data only and do not expose it publicly.
No login, user accounts, or real ISL recognition are implemented.

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
   `clientRequestId`, and `sessionId`. Supported upload signatures are MP4
   (`video/mp4`, `ftyp`) and WebM (`video/webm`, EBML header); the service bounds
   reads to 50 MiB, checks consent/session binding, and does not retain the
   uploaded bytes.
6. Recognition returns `UNSUPPORTED` with a `NEEDS_CLARIFICATION` interpretation,
   no candidates, and `mock-v1` metadata. It deliberately does not pretend to
   process or recognize signs. The patient may request clarification/re-sign or
   send typed text instead.
7. Patient typed/visual messages and doctor-authored messages are persisted at
   `/conversations/{id}/messages`. History requires `session_id` and is paged
   with `offset` and `limit`. Only a patient send action is marked
   `PATIENT_CONFIRMED`; doctor acknowledgement is tracked separately.
8. Confirm/correct/reject endpoints validate the interpretation, patient session,
   consultation, and candidate, then persist confirmation, message, state, and
   audit data in one database transaction. Confirmation requires an
   `ACCEPTABLE` interpretation and candidate; the configured mock never produces
   these, so confirmation is exercised only with isolated synthetic test fixtures.
9. `GET /doctors/consultations` lists active prototype consultations.

All interpretation/recognition reads and message-history requests require the
associated `sessionId` query parameter.
Errors use a standard `code`, `message`, `correlationId`, and optional
`fieldErrors` response body; the same correlation ID is returned in
`X-Correlation-ID`.

## WebSocket notifications

Connect to `/api/v1/ws/consultations/{consultationId}?sessionId={sessionId}`.
The server validates the temporary session and emits minimal identifier-only
notifications after the corresponding database commit, including consultation,
recognition, confirmation, and message events. REST remains authoritative; fetch
REST history after connecting or reconnecting.

Connections and broadcasts are in-memory and process-local. Events are not
replayed after reconnect, server restart, or across multiple API workers; M1 does
not include a transactional outbox.

## Limitations

- M1 has no authentication or production authorization; only synthetic local
  data is appropriate.
- The mock has no video parser/inference. Duration is recorded as unknown, so a
  30-second limit cannot be verified yet; the 50 MiB bound and container
  signature check are not a substitute for full media validation.
- The join-attempt limiter is process-local and cannot protect a multi-worker or
  hosted deployment.
- The doctor consultation list is unauthenticated by design in this prototype.
- No frontend consultation workflow, camera capture, or real-time UI is included;
  those belong to later milestones.

Run backend tests and checks from the repository root as described in the root
README. Start the entire local stack with `docker compose --env-file .env.example
up --build` or use a local `.env` file.
