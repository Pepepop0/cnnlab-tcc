from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import seed_reference_rows


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "tcc.sqlite3"


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def store(db_path: Path, artifact_root: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(db_path, artifact_root)
    yield s
    s.close()


@pytest.fixture
def seeded(store: SQLiteRunStore) -> SQLiteRunStore:
    """Store with protocol, environment, dataset and a frozen synthetic plan."""
    seed_reference_rows(store)
    return store
