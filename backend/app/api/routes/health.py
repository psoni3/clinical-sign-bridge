from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.infrastructure.persistence.database import get_engine
from app.infrastructure.persistence.outbox import read_outbox_health

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> dict[str, object]:
    try:
        database_url = get_settings().database_url.get_secret_value()
        with Session(get_engine(database_url)) as db:
            outbox_health = read_outbox_health(db)
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATABASE_UNAVAILABLE", "message": "Database is not ready"},
        ) from error
    return {
        "status": "ready",
        "notifications": {
            "pending": outbox_health.pending,
            "processing": outbox_health.processing,
            "dead": outbox_health.dead,
            "oldestUnpublishedSeconds": outbox_health.oldest_unpublished_seconds,
        },
    }
