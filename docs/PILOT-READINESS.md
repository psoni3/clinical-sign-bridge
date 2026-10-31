# Pilot Readiness and Residual Risks

## Status

This is an engineering readiness note, not a security certification, privacy/legal
opinion, clinical approval, or permission to use identifiable patient information.
The application remains a local-only prototype for synthetic data. Its no-auth
boundary is intentional; production mode is rejected by backend configuration.

## Prototype threat model

| Asset or boundary | Threat | Current control | Residual risk |
|---|---|---|---|
| Consultation data and join codes | Unauthorized access by someone who obtains a code or session handle | Short-lived consultation workflow, hashed join-code verification, rate limiting, minimal notification events | No user identity or role authorization; anyone holding a valid temporary credential may act within its scope |
| API and browser surfaces | Cross-origin access, host-header abuse, framing or content-type confusion | Explicit CORS origins, trusted-host allowlist, response security headers | No TLS termination in local Compose; controls are not a substitute for production ingress security |
| Database and inference services | Direct access from browser containers or host network | Compose network segmentation; PostgreSQL host binding is loopback-only | Host storage encryption and managed secret handling are not configured by this repository |
| Recognition and confirmation | Incorrect candidate treated as a verified clinical statement | Mock provider abstains; patient confirmation is a server-side invariant; notifications carry identifiers and hints, not message text | No qualified clinical or intended-user validation; model accuracy is not established |
| Notifications | Process crash after a business commit loses a notification; repeated delivery creates duplicate client work | Transactional outbox, lease-based claims, bounded retries, stable event IDs; REST remains authoritative | Delivery is at-least-once, WebSocket connections are process-local, and there is no cross-replica broker or event replay |
| Logs and operational signals | Sensitive payload exposure or unnoticed delivery backlog | Outbox logs contain event IDs, attempt counts, and exception types; readiness exposes aggregate backlog counts and oldest age | No production dashboard, alert routing, centralized log review, or tested SLO is configured |

This threat model is limited to the implemented prototype and has not been independently
reviewed. A formal security review must validate the assumptions and controls before any
pilot.

## Production access-control boundary

Authentication and authorization are not implemented and must not be added as an
implicit milestone shortcut. Before any real-data deployment, the product, security,
privacy, and clinical owners must approve an access-control design covering:

- trusted identity proofing and authentication for each participant type;
- authorization scoped to a consultation and the permitted patient/doctor actions;
- short-lived, revocable session credentials and secure transport;
- audit requirements, failed-access handling, recovery, and account/session revocation;
- deployment ownership, secret management, and operational monitoring.

The current temporary join code and session handle are prototype workflow credentials,
not production access control. Do not expose this service publicly or use it with
identifiable patient information.

## Encryption, retention, and deletion

These controls are **not verified or implemented as production controls**:

- TLS is not terminated by the local Compose stack. Any hosted service requires approved
  TLS termination and protected internal service communication.
- PostgreSQL data-at-rest encryption depends on the host/storage platform and is not
  configured here. Production requires approved managed encryption and key ownership.
- No approved retention duration, deletion schedule, legal hold, or deletion-evidence
  process is defined. Data owners and counsel must set these policies before real data.
- The project does not claim that deleting an application record removes database
  backups, logs, or external copies. A complete deletion workflow must cover each
  storage and backup layer.

Use synthetic data only. Do not upload or retain identifiable patient data.

## Notification reliability and operations

The outbox stores the consultation ID, resource ID, event type, status, timestamps,
attempt count, lease, and error code; it does not store message text or join codes.
Business state and notification intent are written in the same database transaction.
The dispatcher claims work in batches, retries with exponential backoff, and marks
events `DEAD` after ten failed delivery attempts. `/ready` reports pending, processing,
dead, and oldest-unpublished counts without clinical payloads.

Delivery is at-least-once. Clients must treat WebSocket notifications as refresh hints
and re-fetch authoritative state from REST. The current connection manager is
process-local, so multiple API replicas need an approved shared broker/fan-out design
before horizontal scaling. Operators must alert on dead events and increasing backlog;
this repository does not configure alert thresholds or routing.

## Incident response and model rollback

This is a response outline, not an exercised incident plan. Before any pilot, name the
clinical, security, privacy, and operations responders and approve notification,
evidence-retention, and reporting timelines.

For a suspected harmful interpretation, unauthorized access, or data exposure:

1. Stop the affected service or workflow and prevent further use; preserve patient
   access to an appropriate human interpreter or other approved communication route.
2. Notify the designated clinical, security, privacy, and operations owners through
   the approved incident channel.
3. Preserve only the minimum approved audit evidence. Do not copy raw video or
   sensitive payloads into ordinary logs or incident tickets.
4. Assess affected consultations and required notifications under the approved policy.
5. Record the cause, corrective action, and regression tests before resuming the
   affected workflow.

The inference service must remain on its deterministic abstaining mock. No approved
trained model, model registry, deployment approval, or rollback drill exists. If a
future model is introduced, its owner must define a verified prior version, approval
checks, rollback command, and post-rollback validation before enabling it.

## Validation and go/no-go gates

Engineering checks currently cover backend unit/integration behavior, linting, typing,
formatting, and offline migration generation. The bounded load probe is restricted to
loopback `/health` and `/ready`; it is not a clinical workflow benchmark and produces
no capacity claim. PostgreSQL concurrency, target load, alerting, accessibility, and
intended-user workflow evaluation still need evidence.

Do not start a real-patient pilot until all of the following are documented and
approved by the accountable stakeholders:

- production identity, authorization, TLS, encryption, retention, and deletion controls;
- security and privacy/legal reviews and remediation of blocking findings;
- clinical workflow and safety review, including human escalation;
- qualified ISL/Deaf signer engagement, consented data governance, and independent
  evaluation before any real recognition model is considered;
- accessibility and usability evaluation with intended users;
- measured performance and resilience against approved targets;
- incident response and model rollback exercises;
- product, clinical, ISL, security, privacy, and operations go/no-go approval.
