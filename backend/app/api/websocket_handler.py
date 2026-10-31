"""Consultation-scoped, notification-only WebSocket events."""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.helpers.validation import (
    validate_consultation_active,
    validate_session_active,
    validate_session_belongs_to_consultation,
)
from app.config import get_settings
from app.infrastructure.persistence.consultation_repository import (
    ConsultationRepository,
    ConsultationSessionRepository,
)
from app.infrastructure.persistence.database import get_engine

router = APIRouter(prefix="/api/v1", tags=["websocket"])


class ConsultationConnectionManager:
    """Keep local-process sockets and deliver minimal post-commit hints."""

    def __init__(self) -> None:
        self._connections: dict[UUID, set[WebSocket]] = defaultdict(set)

    async def connect(self, consultation_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[consultation_id].add(websocket)

    def disconnect(self, consultation_id: UUID, websocket: WebSocket) -> None:
        connections = self._connections.get(consultation_id)
        if connections is None:
            return
        connections.discard(websocket)
        if not connections:
            self._connections.pop(consultation_id, None)

    async def publish(
        self,
        consultation_id: UUID,
        event_type: str,
        resource_id: UUID,
    ) -> None:
        event = {
            "type": event_type,
            "consultationId": str(consultation_id),
            "resourceId": str(resource_id),
            "occurredAt": datetime.now(UTC).isoformat(),
        }
        for websocket in tuple(self._connections.get(consultation_id, ())):
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(consultation_id, websocket)


connection_manager = ConsultationConnectionManager()


def _validate_connection(
    consultation_id: UUID,
    session_id: UUID,
) -> None:
    database_url = get_settings().database_url.get_secret_value()
    with Session(get_engine(database_url)) as db:
        now = datetime.now(UTC)
        consultation = ConsultationRepository(db).get_by_id(consultation_id)
        consultation = validate_consultation_active(consultation, now)
        session = ConsultationSessionRepository(db).get_by_id(session_id)
        session = validate_session_active(session, now)
        validate_session_belongs_to_consultation(session, consultation_id)
        if session.session_type not in {"PATIENT_DEVICE", "DOCTOR_VIEW"}:
            raise HTTPException(
                status_code=403,
                detail={"code": "SESSION_TYPE_MISMATCH"},
            )


@router.websocket("/ws/consultations/{consultation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    consultation_id: UUID,
    session_id: UUID = Query(..., alias="sessionId"),
) -> None:
    """Connect a temporary session to minimal consultation event notifications."""
    try:
        _validate_connection(consultation_id, session_id)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    except SQLAlchemyError:
        await websocket.close(code=status.WS_1011_SERVER_ERROR)
        return

    await connection_manager.connect(consultation_id, websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except TimeoutError:
                try:
                    _validate_connection(consultation_id, session_id)
                except HTTPException:
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                except SQLAlchemyError:
                    await websocket.close(code=status.WS_1011_SERVER_ERROR)
                    return
    except WebSocketDisconnect:
        return
    finally:
        connection_manager.disconnect(consultation_id, websocket)
