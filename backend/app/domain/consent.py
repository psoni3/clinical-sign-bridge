"""Domain models for consent management."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class ConsentType(StrEnum):
    """Consent purposes (kept separate by spec)."""

    PROCESSING_FOR_LIVE_TRANSLATION = "PROCESSING_FOR_LIVE_TRANSLATION"
    RESEARCH_VIDEO_COLLECTION = "RESEARCH_VIDEO_COLLECTION"


class ConsentStatus(StrEnum):
    """Consent lifecycle."""

    ACTIVE = "ACTIVE"
    WITHDRAWN = "WITHDRAWN"


@dataclass
class Consent:
    """Consent record with purpose, version, and timestamp.

    Keeps live-processing consent (M1) distinct from research consent (future).
    """

    id: UUID
    consultation_id: UUID
    session_id: UUID
    consent_type: ConsentType
    policy_version: str
    accepted: bool
    captured_at: datetime
    withdrawn_at: datetime | None = None

    def is_active(self) -> bool:
        """Check if consent is currently active."""
        return self.accepted and self.withdrawn_at is None
