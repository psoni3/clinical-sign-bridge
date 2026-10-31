"""Security utilities for join codes and session tokens."""

import hashlib
import secrets
from collections import defaultdict, deque
from threading import Lock
from time import monotonic
from uuid import UUID


def generate_join_code() -> str:
    """Generate cryptographically secure join code.

    Format: 12 hexadecimal characters (48 bits of entropy).
    Never log the plaintext value.
    """
    return secrets.token_hex(6).upper()


def generate_session_token() -> str:
    """Generate cryptographically secure session token.

    Format: 32-character hex string.
    Never log the plaintext value.
    """
    return secrets.token_hex(16)


def hash_join_code(plaintext: str) -> str:
    """Hash join code for storage.

    Uses PBKDF2 with SHA-256 and salt.
    """
    salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac("sha256", plaintext.encode("utf-8"), salt, iterations=100_000)
    # Store as "salt$hash" for later verification
    return f"{salt.hex()}${hashed.hex()}"


def hash_session_token(plaintext: str) -> str:
    """Hash session token for storage.

    Uses PBKDF2 with SHA-256 and salt.
    """
    salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac("sha256", plaintext.encode("utf-8"), salt, iterations=100_000)
    return f"{salt.hex()}${hashed.hex()}"


def verify_join_code(plaintext: str, hash_value: str) -> bool:
    """Verify join code matches stored hash.

    Constant-time comparison to prevent timing attacks.
    """
    try:
        salt_hex, stored_hash = hash_value.split("$")
        salt = bytes.fromhex(salt_hex)
        computed = hashlib.pbkdf2_hmac(
            "sha256", plaintext.encode("utf-8"), salt, iterations=100_000
        )
        return secrets.compare_digest(computed.hex(), stored_hash)
    except (ValueError, AttributeError):
        return False


def verify_session_token(plaintext: str, hash_value: str) -> bool:
    """Verify session token matches stored hash.

    Constant-time comparison to prevent timing attacks.
    """
    try:
        salt_hex, stored_hash = hash_value.split("$")
        salt = bytes.fromhex(salt_hex)
        computed = hashlib.pbkdf2_hmac(
            "sha256", plaintext.encode("utf-8"), salt, iterations=100_000
        )
        return secrets.compare_digest(computed.hex(), stored_hash)
    except (ValueError, AttributeError):
        return False


class JoinAttemptLimiter:
    """Single-process fixed-window limiter for the local no-auth prototype."""

    def __init__(self, maximum_attempts: int = 5, window_seconds: int = 60) -> None:
        self._maximum_attempts = maximum_attempts
        self._window_seconds = window_seconds
        self._attempts: dict[tuple[UUID, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, consultation_id: UUID, client: str, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        key = (consultation_id, client)
        cutoff = current - self._window_seconds
        with self._lock:
            attempts = self._attempts[key]
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if len(attempts) >= self._maximum_attempts:
                return False
            attempts.append(current)
            if len(self._attempts) > 8192:
                stale_keys = [
                    stored_key
                    for stored_key, stored_attempts in self._attempts.items()
                    if not stored_attempts or stored_attempts[-1] <= cutoff
                ]
                for stale_key in stale_keys:
                    self._attempts.pop(stale_key, None)
            return True
