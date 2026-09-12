"""
Points the app at a throwaway SQLite file before any backend module is
imported -- database.py binds its engine to config.DATABASE_URL at import
time, so this has to happen before `import database`/`import models`/etc,
which is why it's at module scope in conftest.py rather than inside a
fixture. Tests never touch the real dev rec_fraud.db this way.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP_DB = Path(tempfile.gettempdir()) / "rec_guard_test.db"
if _TMP_DB.exists():
    _TMP_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ.setdefault("SIMULATION_AUTOSTART", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import database  # noqa: E402
import models  # noqa: E402

database.init_db()


@pytest.fixture
def db():
    """One session per test; every table is wiped after, so tests never
    see another test's leftover rows (a fresh run_id per test would work
    too, but a full wipe is simpler and matches what Reset Transactions
    itself does)."""
    session = database.SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        for table in reversed(database.Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()
