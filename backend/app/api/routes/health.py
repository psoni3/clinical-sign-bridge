from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.persistence.database import check_database_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> dict[str, str]:
    try:
        check_database_connection()
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATABASE_UNAVAILABLE", "message": "Database is not ready"},
        ) from error
    return {"status": "ready"}
