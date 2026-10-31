from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings


@lru_cache
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def check_database_connection() -> None:
    database_url = get_settings().database_url.get_secret_value()
    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))
