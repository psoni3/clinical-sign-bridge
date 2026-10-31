"""Database utilities for consistency across tests and app."""

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

# Global reference to keep test engine persistent
_test_engine_cache: dict[str, Engine] = {}


@lru_cache
def get_engine(database_url: str) -> Engine:
    """Get or create database engine.

    For in-memory tests, all calls with same URL return same engine (persistent).
    """
    # Check if we have a cached in-memory engine for this URL
    if ":memory:" in database_url:
        cache_key = database_url
        if cache_key in _test_engine_cache:
            return _test_engine_cache[cache_key]

        # Create new in-memory engine with persistent connection pool
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        _test_engine_cache[cache_key] = engine
        return engine

    # For non-memory databases, create normally
    return create_engine(database_url, pool_pre_ping=True)


def init_test_db() -> None:
    """Initialize in-memory test database schema.

    Call this from conftest before running tests.
    """
    from app.config import get_settings
    from app.infrastructure.persistence.entities import Base

    database_url = get_settings().database_url.get_secret_value()
    engine = get_engine(database_url)

    # Create all tables
    Base.metadata.create_all(engine)


def check_database_connection() -> None:
    """Check database is accessible."""
    from app.config import get_settings

    database_url = get_settings().database_url.get_secret_value()
    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))
