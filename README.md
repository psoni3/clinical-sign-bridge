# Clinical Sign Bridge

An assistive communication prototype foundation for Indian Sign Language (ISL) workflows.
M0 contains two Angular shells, a FastAPI API shell, a separately deployable inference
contract with an abstaining deterministic mock, PostgreSQL/Alembic setup, and local
container orchestration.

> **Prototype only.** No real ISL recognition, consultation workflow, authentication, or
> clinical decision support is implemented. Use synthetic data only. Do not expose this
> no-auth foundation to the public internet or use identifiable patient information.

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
uv run --project backend ruff check app tests migrations
uv run --project backend mypy app
uv run --project backend pip-audit --skip-editable
$env:CLINICAL_DATABASE_URL = "sqlite+pysqlite:///:memory:"
uv run --project backend pytest
$env:CLINICAL_DATABASE_URL = "postgresql+psycopg://"
uv run --project backend alembic upgrade head --sql
uv sync --project ai-engine --group dev
uv run --project ai-engine ruff check app tests
uv run --project ai-engine mypy app
uv run --project ai-engine pip-audit --skip-editable
uv run --project ai-engine pytest
```

The SQLite URL above is for isolated API shell tests only. The PostgreSQL URL without a
host is used only for offline Alembic SQL generation and does not connect to a database.
PostgreSQL migrations are executed by Compose against PostgreSQL. To run the API outside Compose, set
`CLINICAL_DATABASE_URL` to a local PostgreSQL connection URL, then run
`uv run --project backend alembic upgrade head` and
`uv run --project backend uvicorn app.main:app --reload`.

To run the inference shell outside Compose:

```powershell
uv run --project ai-engine uvicorn app.main:app --reload --port 8001
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

M0 provides startup foundations only. Consultation/session lifecycle, consent APIs,
conversation behavior, recognition orchestration, confirmation transactions, and
WebSocket notifications are M1 work. Camera capture, patient confirmation UI, and doctor
conversation workflows are later milestones. No authentication feature is present.

Read [context/00-COPILOT-INSTRUCTIONS.md](context/00-COPILOT-INSTRUCTIONS.md) and
[context/10-DELIVERY-PLAN.md](context/10-DELIVERY-PLAN.md) before extending this prototype.
