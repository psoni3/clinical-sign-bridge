import os

os.environ["CLINICAL_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

# Initialize database schema before any imports that use the db
from app.infrastructure.persistence.database import init_test_db

init_test_db()
