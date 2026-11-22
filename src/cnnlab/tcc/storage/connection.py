"""SQLite connections with the durability settings required by TCC-04.

Every writer connection runs with `foreign_keys=ON`, `journal_mode=WAL` and
`synchronous=FULL`; the pragmas are read back and a mismatch aborts the open
instead of silently running without the guarantee. Readers are opened with
`mode=ro` plus `query_only`, so an analysis tool cannot write by accident.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# STRICT tables need 3.37; built-in json_valid needs 3.38.
SQLITE_MIN_VERSION: tuple[int, int, int] = (3, 38, 0)
BUSY_TIMEOUT_MS = 10_000
_SYNCHRONOUS_FULL = 2


def _check_library_version() -> None:
    if sqlite3.sqlite_version_info < SQLITE_MIN_VERSION:
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} is older than required {'.'.join(map(str, SQLITE_MIN_VERSION))}"
        )


def _require_file_path(path: Path) -> Path:
    text = str(path)
    if not text or text == ":memory:" or text.startswith("file:"):
        raise ValueError("a filesystem path is required (WAL is unavailable in memory)")
    return Path(path)


def open_writer(path: Path) -> sqlite3.Connection:
    """Open the single writer connection; transactions are managed explicitly."""
    _check_library_version()
    path = _require_file_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,
        check_same_thread=False,
        timeout=BUSY_TIMEOUT_MS / 1000,
    )
    try:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        _verify(conn, expect_wal=True)
        if not is_synchronous_full(conn):
            raise RuntimeError("PRAGMA synchronous could not be set to FULL")
    except BaseException:
        conn.close()
        raise
    return conn


def open_reader(path: Path) -> sqlite3.Connection:
    """Read-only connection for queries/analysis; cannot write even by mistake."""
    _check_library_version()
    path = _require_file_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
    try:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA query_only = ON")
        _verify(conn, expect_wal=True)
    except BaseException:
        conn.close()
        raise
    return conn


def _verify(conn: sqlite3.Connection, *, expect_wal: bool) -> None:
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("PRAGMA foreign_keys could not be enabled")
    if expect_wal:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            raise RuntimeError(f"journal_mode is {mode!r}, expected 'wal'")


def pragma_snapshot(conn: sqlite3.Connection) -> dict[str, object]:
    """Effective durability pragmas, for diagnostics and tests."""
    return {
        "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
        "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
        "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
        "busy_timeout": conn.execute("PRAGMA busy_timeout").fetchone()[0],
    }


def is_synchronous_full(conn: sqlite3.Connection) -> bool:
    return conn.execute("PRAGMA synchronous").fetchone()[0] == _SYNCHRONOUS_FULL
