# Clinical Sign Bridge

An assistive communication prototype for Indian Sign Language (ISL) workflows. M0
provides two Angular shells, a FastAPI foundation, a replaceable inference contract with
an abstaining deterministic mock, PostgreSQL/Alembic setup, and local container
orchestration. M1 adds the consultation and conversation backend. M2 adds the patient
consultation, consent, camera capture, interpretation review, and messaging workflow.

> **Prototype only.** No real ISL recognition, production authentication, or clinical
> decision support is implemented. The M1 consultation workflow is a local backend
> prototype. Use synthetic data only. Do not expose this no-auth foundation to the public
> internet or use identifiable patient information.

## Local development with Docker Compose

Prerequisites: Docker Desktop with the Linux container engine, Node.js 22.12+ and npm 10+
for frontend-only development, and `uv` plus Python 3.12 for backend-only development.

1. Copy `.env.example` to `.env` and set a unique local-only `POSTGRES_PASSWORD` using
   URL-safe letters, digits, and hyphens. Keep `.env` out of source control.
2. From the repository root, run:

   ```powershell
   docker compose up --build
   ```

Compose starts PostgreSQL, runs the Alembic migration before the API, and serves the
patient and doctor shells. PostgreSQL and API health checks gate dependent services.

| Component | Address |
|---|---|
| Patient shell | http://localhost:4200 |
| Doctor shell | http://localhost:4201 |
| Backend OpenAPI | http://localhost:8000/docs |
| Backend health / readiness | http://localhost:8000/health and `/ready` |
| Mock inference OpenAPI | http://localhost:8001/docs |
| Inference health / readiness | http://localhost:8001/health and `/ready` |

Stop the stack with `Ctrl+C`. The PostgreSQL volume persists across restarts.

## Run checks locally

From the repository root:

```powershell
cd frontend
npm ci --legacy-peer-deps
npm run lint
npm test
npm run build:patient
npm run build:doctor
npm audit --audit-level=high
cd ..
uv sync --project backend --group dev
uv run --project backend ruff check backend\app backend\tests backend\migrations
uv run --project backend mypy backend\app
uv run --project backend pip-audit --skip-editable
$env:CLINICAL_DATABASE_URL = "sqlite+pysqlite:///:memory:"
uv run --project backend pytest backend\tests -q
Set-Location backend
$env:CLINICAL_DATABASE_URL = "postgresql+psycopg://user:password@localhost/clinical"
uv run alembic upgrade head --sql
Set-Location ..
uv sync --project ai-engine --group dev
uv run --project ai-engine ruff check ai-engine\app ai-engine\tests
uv run --project ai-engine mypy ai-engine\app
uv run --project ai-engine pip-audit --skip-editable
uv run --project ai-engine pytest ai-engine\tests -q
```

The SQLite URL above is for isolated API shell tests only. The PostgreSQL URL without a
host is not needed for offline Alembic SQL generation and does not connect to a database.
PostgreSQL migrations are executed by Compose against PostgreSQL. To run the API outside
Compose, set `CLINICAL_DATABASE_URL` to a local PostgreSQL connection URL, then run these
commands from the repository root:

```powershell
Set-Location backend
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

To run the inference shell outside Compose:

```powershell
Set-Location ai-engine
uv run uvicorn app.main:app --reload --port 8001
```

The mock provider always returns `UNSUPPORTED` with no candidates. It does not inspect
video or recognize signs. This deliberate abstention avoids suggesting that mocked text
is a valid translation.

## Repository layout

```text
ai-engine/        Inference schemas, replaceable provider contract, mock service and tests
backend/          FastAPI shell, SQLAlchemy model baseline, Alembic and tests
context/          Product, architecture, safety and delivery specifications
frontend/         Angular workspace with separate patient and doctor applications
.github/workflows/ CI checks for frontend, backend, inference and migration SQL
docker-compose.yml Local PostgreSQL, API, inference and frontend services
```

## Scope and next milestone

M0 foundation, M1 consultation/conversation backend, and the M2 patient workflow are
implemented, along with the M3 doctor dashboard. Clinicians can create and start
consultations, share the generated one-time join code, view patient-confirmed messages,
acknowledge them as clinician-reviewed, ask questions, and complete consultations.
Patient users join with a consultation ID and short code; code-only lookup is not
available. The doctor dashboard refreshes history from REST after WebSocket notifications
and supports read-only completed history. The patient view also switches to read-only when
the consultation completes or expires, and stops reconnecting to its notification channel.

M2 supports explicit processing consent, optional camera capture (up to 30 seconds),
patient-reviewed interpretations, typed and visual responses, and an interpreter-request
message. An interpreter request is recorded in the conversation but does not book an
interpreter. The 30-second capture limit is enforced in the patient browser only; the
backend currently enforces the upload-size bound but does not parse duration.

The mock always abstains: it does not inspect video or recognize ISL, and produces no
candidates. Neither interface must be used with real patient information. No
authentication feature is present; anyone with a consultation code can join, and the
temporary session handles are not production access control. Automated cross-browser
end-to-end coverage remains outstanding. Backend/API details and prototype limitations are in
[backend/M1-README.md](backend/M1-README.md).

Patient and doctor Angular component/API-client tests, production builds, backend
integration tests, and frontend formatting checks cover the implemented workflows.
They do not replace automated browser-level end-to-end testing against the running
services.

M4 introduces dataset manifest validation/versioning and signer-separated split
tooling, plus optional MediaPipe preprocessing and a temporal gloss-classification
baseline. These are engineering tools only: governance and annotation materials are
drafts awaiting qualified review, no signer data or trained model is included, and the
inference service continues to default to the abstaining mock. See
[ai-engine/README.md](ai-engine/README.md) and
[ai-engine/DATASET-GOVERNANCE.md](ai-engine/DATASET-GOVERNANCE.md).

Read [context/00-COPILOT-INSTRUCTIONS.md](context/00-COPILOT-INSTRUCTIONS.md) and
[context/10-DELIVERY-PLAN.md](context/10-DELIVERY-PLAN.md) before extending this prototype.

## M5 pilot-hardening status

M5 adds a transactional notification outbox, explicit local-only deployment safeguards,
container privilege/network restrictions, and a loopback-only readiness probe. The
outbox uses at-least-once delivery; WebSocket notifications are refresh hints, and REST
remains authoritative. See [docs/PILOT-READINESS.md](docs/PILOT-READINESS.md) for the
threat model, deployment boundary, incident response outline, and open approval gates.

M5 is **not complete** and this application is **not pilot-ready**. Authentication and
authorization, TLS, verified encryption at rest, approved retention/deletion policy,
production monitoring/alerting, target-load evaluation, accessibility/usability
validation, incident drills, and formal stakeholder approvals remain outstanding.
Keep deployment local and use synthetic data only.

With the local stack running, the bounded readiness probe can be run from the repository
root:

```powershell
uv run --project backend python backend\scripts\load_test.py --target http://127.0.0.1:8000/ready --requests 100 --concurrency 10
```

The probe only accepts loopback `/health` or `/ready` URLs. It does not measure
consultation workflow capacity or establish a production performance target.
