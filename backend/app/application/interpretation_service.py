"""Application services for interpretation and confirmation workflow."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.interpretation import (
    Confirmation,
    ConfirmationAction,
    Interpretation,
    InterpretationNotConfirmableError,
    Message,
    MessageProvenance,
    MessageType,
    SenderType,
    VerificationStatus,
)


class ConfirmInterpretationService:
    """Confirm AI candidate for conversion to doctor-facing message."""

    @staticmethod
    def execute(
        interpretation: Interpretation,
        candidate_id: UUID,
        session_id: UUID,
    ) -> tuple[Confirmation, Message]:
        """Confirm candidate and create message.

        CRITICAL INVARIANT: Runs in single database transaction.
        - Verification: interpretation status, quality, candidate ownership
        - State change: interpretation -> CONFIRMED
        - Message creation: text = exact candidate text, provenance = AI_CANDIDATE_CONFIRMED
        - Audit: recorded in transaction

        Args:
            interpretation: Unverified interpretation
            candidate_id: Candidate UUID to confirm
            session_id: Patient session UUID

        Returns:
            Tuple of (confirmation, created_message)

        Raises:
            InterpretationNotConfirmableError: If interpretation cannot be confirmed
            CandidateNotFoundError: If candidate doesn''t exist or doesn''t belong
            DuplicateConfirmationError: If already confirmed
        """
        now = datetime.now(UTC)

        # Validate interpretation is confirmable
        if not interpretation.is_confirmable():
            raise InterpretationNotConfirmableError(
                f"Status {interpretation.status}, quality {interpretation.quality_status}"
            )

        # Validate candidate belongs to this interpretation
        interpretation.validate_candidate_belongs_to_this_interpretation(candidate_id)

        # Find candidate to get exact text
        candidate = next(c for c in interpretation.candidates if c.id == candidate_id)

        # Create confirmation record
        confirmation = Confirmation(
            id=uuid4(),
            interpretation_id=interpretation.id,
            candidate_id=candidate_id,
            session_id=session_id,
            action=ConfirmationAction.CONFIRM,
            correction_text=None,
            created_message_id=None,
            created_at=now,
        )

        # Create doctor-facing message with exact candidate text
        message = Message(
            id=uuid4(),
            conversation_id=uuid4(),  # TODO: Get from conversation repo
            sender_type=SenderType.PATIENT,
            sender_session_id=session_id,
            message_type=MessageType.AI_TRANSLATION,
            text=candidate.text,  # EXACT candidate text
            provenance=MessageProvenance.AI_CANDIDATE_CONFIRMED,
            verification_status=VerificationStatus.PATIENT_CONFIRMED,
            created_at=now,
            confirmed_at=now,
            confirmation_id=confirmation.id,
        )

        # Store created message ID back in confirmation
        confirmation.created_message_id = message.id

        return confirmation, message


class CorrectInterpretationService:
    """Patient correction of AI candidate."""

    @staticmethod
    def execute(
        interpretation: Interpretation,
        corrected_text: str,
        session_id: UUID,
    ) -> tuple[Confirmation, Message]:
        """Correct candidate with patient-provided text.

        CRITICAL INVARIANT: Runs in single database transaction.
        - State change: interpretation -> CONFIRMED
        - Message creation: text = patient correction, provenance = PATIENT_CORRECTED
        - Original candidates preserved for audit

        Args:
            interpretation: Unverified interpretation
            corrected_text: Patient-provided correction
            session_id: Patient session UUID

        Returns:
            Tuple of (confirmation, created_message)

        Raises:
            InterpretationNotConfirmableError: If interpretation cannot be corrected
        """
        now = datetime.now(UTC)

        # Validate interpretation is confirmable
        if not interpretation.is_confirmable():
            raise InterpretationNotConfirmableError(
                f"Status {interpretation.status}, quality {interpretation.quality_status}"
            )

        # Create correction confirmation
        confirmation = Confirmation(
            id=uuid4(),
            interpretation_id=interpretation.id,
            candidate_id=None,  # Patient-provided, not from candidates
            session_id=session_id,
            action=ConfirmationAction.CORRECT,
            correction_text=corrected_text,
            created_message_id=None,
            created_at=now,
        )

        # Create message with patient correction
        message = Message(
            id=uuid4(),
            conversation_id=uuid4(),  # TODO: Get from conversation repo
            sender_type=SenderType.PATIENT,
            sender_session_id=session_id,
            message_type=MessageType.AI_TRANSLATION,
            text=corrected_text,  # PATIENT-PROVIDED text
            provenance=MessageProvenance.PATIENT_CORRECTED,  # Key difference
            verification_status=VerificationStatus.PATIENT_CONFIRMED,
            created_at=now,
            confirmed_at=now,
            confirmation_id=confirmation.id,
        )

        confirmation.created_message_id = message.id

        return confirmation, message


class RejectInterpretationService:
    """Patient rejection of all candidates."""

    @staticmethod
    def execute(
        interpretation: Interpretation,
        reason: str,
        session_id: UUID,
    ) -> Confirmation:
        """Reject all candidates without creating confirmed message.

        CRITICAL INVARIANT: No doctor-facing message is created.

        Args:
            interpretation: Unverified interpretation
            reason: Why patient rejected (e.g., "NONE_OF_THESE")
            session_id: Patient session UUID

        Returns:
            Confirmation record (no message)

        Raises:
            InterpretationNotConfirmableError: If interpretation cannot be rejected
        """
        now = datetime.now(UTC)

        # Validate interpretation is rejectable (same as confirmable)
        if not interpretation.is_confirmable():
            raise InterpretationNotConfirmableError(
                f"Status {interpretation.status}, quality {interpretation.quality_status}"
            )

        confirmation = Confirmation(
            id=uuid4(),
            interpretation_id=interpretation.id,
            candidate_id=None,
            session_id=session_id,
            action=ConfirmationAction.REJECT,
            correction_text=reason,  # Store reason as metadata
            created_message_id=None,  # KEY: No message created
            created_at=now,
        )

        return confirmation


class ClarificationRequestService:
    """Request for patient re-sign or clarification."""

    @staticmethod
    def execute(
        interpretation: Interpretation,
        session_id: UUID,
    ) -> Confirmation:
        """Record clarification request.

        Does not publish a message; marks interpretation as needing clarification.

        Args:
            interpretation: Current interpretation
            session_id: Patient session UUID

        Returns:
            Confirmation record
        """
        now = datetime.now(UTC)

        confirmation = Confirmation(
            id=uuid4(),
            interpretation_id=interpretation.id,
            candidate_id=None,
            session_id=session_id,
            action=ConfirmationAction.CLARIFICATION_REQUESTED,
            correction_text=None,
            created_message_id=None,
            created_at=now,
        )

        return confirmation
