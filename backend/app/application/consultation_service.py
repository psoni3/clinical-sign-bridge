"""Application services for consultation workflow."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.domain.consultation import (
    Consultation,
    ConsultationSession,
    JoinCode,
    SessionToken,
)
from app.domain.security import (
    generate_join_code,
    generate_session_token,
    verify_join_code,
)


class CreateConsultationService:
    """Create a new consultation with join code."""

    @staticmethod
    def execute(
        display_label: str | None,
        expires_in_minutes: int,
    ) -> tuple[Consultation, JoinCode]:
        """Create consultation with temporary join code.

        Args:
            display_label: Optional display name for consultation
            expires_in_minutes: How long until consultation expires

        Returns:
            Tuple of (consultation entity, join code value object)

        Note: Join code is never stored plaintext. Only hash is persisted.
        """
        now = datetime.now(UTC)
        consultation_id = uuid4()
        expires_at = now + timedelta(minutes=expires_in_minutes)
        join_code_plaintext = generate_join_code()

        consultation = Consultation(
            id=consultation_id,
            display_label=display_label,
            status="CREATED",  # Use string, not enum
            expires_at=expires_at,
            created_at=now,
        )

        join_code = JoinCode(
            plaintext=join_code_plaintext,
            expires_at=expires_at,
        )

        return consultation, join_code


class JoinConsultationService:
    """Join consultation with code and create session."""

    @staticmethod
    def execute(
        consultation: Consultation,
        join_code_plaintext: str,
        join_code_hash: str,
        device_label: str | None,
        session_type: str = "PATIENT_DEVICE",
        session_duration_minutes: int = 120,
    ) -> tuple[ConsultationSession, SessionToken]:
        """Join consultation and create session.

        Args:
            consultation: Consultation entity
            join_code_plaintext: Plaintext code provided by patient
            join_code_hash: Stored hash to verify
            device_label: Optional device identifier
            session_duration_minutes: Session validity duration

        Returns:
            Tuple of (session entity, session token value object)

        Raises:
            ValueError: If join code does not match hash
        """
        now = datetime.now(UTC)

        # Validate join code
        if not verify_join_code(join_code_plaintext, join_code_hash):
            raise ValueError("Invalid join code")

        # Create session (patient device view)
        session_id = uuid4()
        session_expires_at = now + timedelta(minutes=session_duration_minutes)
        session_token_plaintext = generate_session_token()

        session = ConsultationSession(
            id=session_id,
            consultation_id=consultation.id,
            session_type=session_type,
            device_label=device_label,
            created_at=now,
            expires_at=session_expires_at,
        )

        token = SessionToken(
            plaintext=session_token_plaintext,
            expires_at=session_expires_at,
            session_type=session_type,
        )

        return session, token


class StartConsultationService:
    """Transition consultation from JOINED to ACTIVE."""

    @staticmethod
    def execute(consultation: Consultation) -> Consultation:
        """Start consultation when doctor joins.

        Args:
            consultation: Consultation in JOINED or CREATED state

        Returns:
            Updated consultation with ACTIVE status and started_at timestamp
        """
        now = datetime.now(UTC)

        # Validate can transition
        if consultation.status not in ("CREATED", "JOINED"):
            raise ValueError(f"Cannot start consultation in {consultation.status} state")

        consultation.status = "ACTIVE"
        consultation.started_at = now

        return consultation


class CompleteConsultationService:
    """Transition consultation to COMPLETED."""

    @staticmethod
    def execute(consultation: Consultation) -> Consultation:
        """Complete consultation (no new messages/recognitions).

        Args:
            consultation: Active consultation

        Returns:
            Updated consultation with COMPLETED status and completed_at timestamp
        """
        now = datetime.now(UTC)

        consultation.validate_active_for_operation(now)
        consultation.status = "COMPLETED"
        consultation.completed_at = now

        return consultation


class CreateDoctorSessionService:
    """Create doctor/viewer session for consultation (M1 prep for M3)."""

    @staticmethod
    def execute(
        consultation: Consultation,
        session_duration_minutes: int = 120,
    ) -> tuple[ConsultationSession, SessionToken]:
        """Create doctor viewer session.

        Args:
            consultation: Active consultation
            session_duration_minutes: Session validity duration

        Returns:
            Tuple of (session entity, session token value object)
        """
        now = datetime.now(UTC)

        session_id = uuid4()
        session_expires_at = now + timedelta(minutes=session_duration_minutes)
        session_token_plaintext = generate_session_token()

        session = ConsultationSession(
            id=session_id,
            consultation_id=consultation.id,
            session_type="DOCTOR_VIEW",  # Use string, not enum
            device_label="doctor_web",
            created_at=now,
            expires_at=session_expires_at,
        )

        token = SessionToken(
            plaintext=session_token_plaintext,
            expires_at=session_expires_at,
            session_type="DOCTOR_VIEW",  # Use string, not enum
        )

        return session, token
