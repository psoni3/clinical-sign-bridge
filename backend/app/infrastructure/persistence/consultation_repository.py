"""Repository layer for consultation persistence."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.domain.consultation import Consultation, ConsultationSession
from app.domain.security import hash_join_code, hash_session_token
from app.infrastructure.persistence.entities import (
    Consultation as ConsultationEntity,
)
from app.infrastructure.persistence.entities import (
    ConsultationSession as ConsultationSessionEntity,
)


def _make_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class ConsultationRepository:
    """Repository for Consultation aggregates."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, consultation: Consultation) -> None:
        """Update lifecycle fields without overwriting the stored join-code hash."""
        entity = self.session.get(ConsultationEntity, consultation.id)
        if entity is None:
            raise ValueError("Consultation must be persisted before it can be updated")
        entity.display_label = consultation.display_label
        entity.status = consultation.status
        entity.expires_at = consultation.expires_at
        entity.created_at = consultation.created_at
        entity.started_at = consultation.started_at
        entity.completed_at = consultation.completed_at

    def save_with_join_code(
        self,
        consultation: Consultation,
        join_code_plaintext: str,
    ) -> None:
        """Persist consultation with join code hash."""
        join_code_hash = hash_join_code(join_code_plaintext)
        entity = ConsultationEntity(
            id=consultation.id,
            display_label=consultation.display_label,
            status=consultation.status,  # status is string in domain
            join_code_hash=join_code_hash,
            expires_at=consultation.expires_at,
            created_at=consultation.created_at,
            started_at=consultation.started_at,
            completed_at=consultation.completed_at,
        )
        self.session.add(entity)

    def get_by_id(self, consultation_id: UUID) -> Consultation | None:
        """Fetch consultation by ID."""
        entity = self.session.execute(
            select(ConsultationEntity).where(ConsultationEntity.id == consultation_id)
        ).scalar_one_or_none()

        if not entity:
            return None

        return Consultation(
            id=entity.id,
            display_label=entity.display_label,
            status=entity.status,  # Already string in DB
            expires_at=_make_aware(entity.expires_at) or datetime.now(UTC),
            created_at=_make_aware(entity.created_at) or datetime.now(UTC),
            started_at=_make_aware(entity.started_at),
            completed_at=_make_aware(entity.completed_at),
        )

    def get_join_code_hash(self, consultation_id: UUID) -> str | None:
        """Get stored join code hash."""
        entity = self.session.execute(
            select(ConsultationEntity.join_code_hash).where(
                ConsultationEntity.id == consultation_id
            )
        ).scalar_one_or_none()
        return entity

    def list_active(self, now: datetime | None = None) -> list[Consultation]:
        """List consultations that are active (not expired/completed)."""
        now = now or datetime.now(UTC)

        entities = (
            self.session.execute(
                select(ConsultationEntity).where(
                    and_(
                        ConsultationEntity.status.in_(["CREATED", "JOINED", "ACTIVE"]),
                        ConsultationEntity.expires_at > now,
                    )
                )
            )
            .scalars()
            .all()
        )

        return [
            Consultation(
                id=e.id,
                display_label=e.display_label,
                status=e.status,
                expires_at=_make_aware(e.expires_at) or datetime.now(UTC),
                created_at=_make_aware(e.created_at) or datetime.now(UTC),
                started_at=_make_aware(e.started_at),
                completed_at=_make_aware(e.completed_at),
            )
            for e in entities
        ]


class ConsultationSessionRepository:
    """Repository for ConsultationSession aggregates."""

    def __init__(self, session: Session):
        self.session = session

    def save(
        self,
        session_obj: ConsultationSession,
        session_token_plaintext: str,
    ) -> None:
        """Persist session with token hash."""
        token_hash = hash_session_token(session_token_plaintext)
        entity = ConsultationSessionEntity(
            id=session_obj.id,
            consultation_id=session_obj.consultation_id,
            session_type=session_obj.session_type,
            session_token_hash=token_hash,
            device_label=session_obj.device_label,
            created_at=session_obj.created_at,
            expires_at=session_obj.expires_at,
        )
        self.session.add(entity)

    def get_by_id(self, session_id: UUID) -> ConsultationSession | None:
        """Fetch session by ID."""
        entity = self.session.execute(
            select(ConsultationSessionEntity).where(ConsultationSessionEntity.id == session_id)
        ).scalar_one_or_none()

        if not entity:
            return None

        return ConsultationSession(
            id=entity.id,
            consultation_id=entity.consultation_id,
            session_type=entity.session_type,
            device_label=entity.device_label,
            created_at=_make_aware(entity.created_at) or datetime.now(UTC),
            expires_at=_make_aware(entity.expires_at) or datetime.now(UTC),
            revoked_at=_make_aware(entity.revoked_at),
            last_seen_at=_make_aware(entity.last_seen_at),
        )

    def get_session_token_hash(self, session_id: UUID) -> str | None:
        """Get stored session token hash."""
        entity = self.session.execute(
            select(ConsultationSessionEntity.session_token_hash).where(
                ConsultationSessionEntity.id == session_id
            )
        ).scalar_one_or_none()
        return entity

    def list_for_consultation(
        self,
        consultation_id: UUID,
        now: datetime | None = None,
    ) -> list[ConsultationSession]:
        """List active sessions for consultation."""
        now = now or datetime.now(UTC)

        entities = (
            self.session.execute(
                select(ConsultationSessionEntity).where(
                    and_(
                        ConsultationSessionEntity.consultation_id == consultation_id,
                        ConsultationSessionEntity.expires_at > now,
                        ConsultationSessionEntity.revoked_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        return [
            ConsultationSession(
                id=e.id,
                consultation_id=e.consultation_id,
                session_type=e.session_type,
                device_label=e.device_label,
                created_at=_make_aware(e.created_at) or datetime.now(UTC),
                expires_at=_make_aware(e.expires_at) or datetime.now(UTC),
                revoked_at=_make_aware(e.revoked_at),
                last_seen_at=_make_aware(e.last_seen_at),
            )
            for e in entities
        ]
