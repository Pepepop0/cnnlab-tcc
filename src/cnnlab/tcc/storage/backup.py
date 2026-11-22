"""Consistent backup of the SQLite store and its artifacts, plus restore rehearsal.

Backup (`create_backup`):

1. the database is copied with SQLite's online backup API from a read-only
   connection, in a single step, so the copy is one consistent WAL snapshot
   even while the worker keeps committing; the copy is switched to
   `journal_mode=DELETE` so it is one self-contained file;
2. the artifact catalog is read **from that copy**, i.e. exactly the files the
   snapshot references; every final file under the artifact root is then
   copied and hashed once. Files are classified, never repaired:
   `verified`, `mismatched` (hash differs from the catalog), `orphan` (not
   in the snapshot catalog, e.g. published after it or before a crash),
   `missing` (cataloged, absent on disk) and `temporary` (publication in
   progress or interrupted; listed with its size, not copied, because
   holding it open on Windows would break the worker's `os.replace`);
3. a manifest (`tcc-backup/1`) records schema versions, the database hash,
   `integrity_check`, row count and content digest of every table, and the
   artifact entries; its own SHA-256 goes to `MANIFEST.sha256`;
4. everything is written to a hidden staging directory and renamed into
   place at the end. An existing destination is refused, never overwritten.

The source database and artifact root are only read. Restore goes to new,
empty targets only; `rehearse_restore` does it in a scratch directory and
checks that the restored store opens, passes its integrity checks and holds
the same table digests and catalog as the manifest.

What this does not prove: that the copy survives a power cut or a failing
disk. Files are fsynced; on Windows directory entries cannot be fsynced.
Keep backups on a different drive and rehearse restores periodically.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from cnnlab.tcc.storage.artifacts import TEMP_SUFFIX
from cnnlab.tcc.storage.canonical import canonical_json, format_utc, sha256_hex, utc_now
from cnnlab.tcc.storage.errors import BackupError
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

BACKUP_FORMAT = "tcc-backup/1"
DB_FILENAME = "tcc.sqlite3"
ARTIFACTS_DIRNAME = "artifacts"
MANIFEST_FILENAME = "MANIFEST.json"
MANIFEST_SHA_FILENAME = "MANIFEST.sha256"
_CHUNK = 1024 * 1024

ArtifactStatus = Literal["verified", "mismatched", "orphan", "temporary", "missing"]


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    rel_path: str
    status: ArtifactStatus
    sha256: str | None  # of the bytes copied (None when missing, or a temporary that is not copied)
    size_bytes: int | None
    cataloged_sha256: str | None  # what the snapshot catalog says (None for orphans/temporaries)


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    artifacts: tuple[ArtifactEntry, ...]

    @property
    def complete(self) -> bool:
        """True when every cataloged artifact was present with its cataloged hash."""
        return bool(self.manifest["complete"])


@dataclass(frozen=True, slots=True)
class BackupVerification:
    path: Path
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class RestoreReport:
    db_path: Path
    artifact_root: Path
    backup_complete: bool
    integrity_messages: tuple[str, ...]
    foreign_key_violations: int
    invariant_problems: tuple[str, ...]
    table_mismatches: tuple[str, ...]
    catalog_matches: bool
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    orphans: tuple[str, ...]
    temporaries: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Restored data equals the backup and passes every integrity check.

        Orphans and temporaries are evidence carried over as-is, reported but
        not a restore failure; missing/mismatched artifacts are failures.
        """
        return (
            self.integrity_messages == ("ok",)
            and self.foreign_key_violations == 0
            and not self.invariant_problems
            and not self.table_mismatches
            and self.catalog_matches
            and not self.missing
            and not self.mismatched
        )


# -- low-level helpers ---------------------------------------------------------


def _fsync_file(path: Path) -> None:
    with open(path, "r+b") as fh:  # Windows needs a writable handle to flush to disk
        os.fsync(fh.fileno())


def _copy_hashing(src: Path, dst: Path) -> tuple[str, int]:
    """Copy `src` to a new `dst` (never overwrites), fsync it, return hash and size of the bytes copied."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        while chunk := fin.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _ro_uri(path: Path, *, immutable: bool = False) -> str:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return uri + "&immutable=1" if immutable else uri


def _is_within(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def table_digests(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Row count and SHA-256 of the canonical content of every table (rowid order)."""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    out: dict[str, dict[str, Any]] = {}
    for name in tables:
        digest = hashlib.sha256()
        rows = 0
        for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid'):
            digest.update(canonical_json(list(row)).encode("utf-8"))
            digest.update(b"\n")
            rows += 1
        out[name] = {"rows": rows, "sha256": digest.hexdigest()}
    return out


def _walk_files(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    if not root.exists():
        return found
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            found.append((full.relative_to(root).as_posix(), full))
    return sorted(found)


# -- backup --------------------------------------------------------------------


def _snapshot_database(db_path: Path, target: Path) -> None:
    src = sqlite3.connect(_ro_uri(db_path), uri=True)
    try:
        src.execute("PRAGMA query_only = ON")
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)  # pages=-1: one step, one read transaction, one snapshot
            dst.execute("PRAGMA journal_mode = DELETE")
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    for suffix in ("-wal", "-shm"):
        if Path(str(target) + suffix).exists():
            raise BackupError(f"snapshot left a {suffix} file; the copy is not self-contained")
    _fsync_file(target)


def _describe_database(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(_ro_uri(path, immutable=True), uri=True)
    try:
        integrity = [r[0] for r in conn.execute("PRAGMA integrity_check")]
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        versions = [
            list(r) for r in conn.execute("SELECT version, name, sql_sha256 FROM schema_migrations ORDER BY version")
        ]
        catalog = {r[0]: r[1] for r in conn.execute("SELECT rel_path, sha256 FROM artifacts")}
        tables = table_digests(conn)
    finally:
        conn.close()
    sha, size = _sha256_file(path)
    return {
        "file": DB_FILENAME,
        "sha256": sha,
        "size_bytes": size,
        "integrity_check": integrity,
        "foreign_key_violations": len(fk),
        "schema_versions": versions,
        "tables": tables,
        "_catalog": catalog,
    }


def create_backup(
    db_path: Path,
    artifact_root: Path,
    dest_dir: Path,
    *,
    label: str,
    clock: Callable[[], datetime] = utc_now,
) -> BackupResult:
    """Write a new backup directory `dest_dir`; the sources are only read."""
    db_path, artifact_root, dest_dir = Path(db_path), Path(artifact_root), Path(dest_dir)
    if not label:
        raise ValueError("a backup needs a label")
    if not db_path.is_file():
        raise BackupError(f"database not found: {db_path}")
    if dest_dir.exists():
        raise BackupError(f"backup destination already exists, refusing to overwrite: {dest_dir}")
    if _is_within(dest_dir, artifact_root):
        raise BackupError("backup destination must be outside the artifact root")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = dest_dir.parent / f".{dest_dir.name}.partial-{secrets.token_hex(4)}"
    staging.mkdir()
    try:
        created_at = format_utc(clock())
        db_copy = staging / DB_FILENAME
        _snapshot_database(db_path, db_copy)
        database = _describe_database(db_copy)
        catalog: dict[str, str] = database.pop("_catalog")
        by_key = {path.casefold(): path for path in catalog}

        entries: list[ArtifactEntry] = []
        seen: set[str] = set()
        vanished: list[str] = []
        for rel, full in _walk_files(artifact_root):
            name = PurePosixPath(rel).name
            if name.startswith(".") and name.endswith(TEMP_SUFFIX):
                # Listed, not copied: a live one belongs to a publication in progress, and
                # holding it open on Windows would make the worker's os.replace fail.
                try:
                    size = full.stat().st_size
                except FileNotFoundError:
                    vanished.append(rel)
                    continue
                entries.append(ArtifactEntry(rel, "temporary", None, size, None))
                continue
            try:
                sha, size = _copy_hashing(full, staging / ARTIFACTS_DIRNAME / PurePosixPath(rel))
            except FileNotFoundError:
                vanished.append(rel)  # a cataloged file that vanished is reported as missing below
                continue
            key = rel.casefold()
            if key not in by_key:
                entries.append(ArtifactEntry(rel, "orphan", sha, size, None))
                continue
            seen.add(key)
            expected = catalog[by_key[key]]
            status: ArtifactStatus = "verified" if sha == expected else "mismatched"
            entries.append(ArtifactEntry(rel, status, sha, size, expected))
        for key, path in sorted(by_key.items()):
            if key not in seen:
                entries.append(ArtifactEntry(path, "missing", None, None, catalog[path]))
        entries.sort(key=lambda e: e.rel_path)

        counts = {
            s: sum(1 for e in entries if e.status == s)
            for s in ("verified", "mismatched", "orphan", "temporary", "missing")
        }
        complete = (
            database["integrity_check"] == ["ok"]
            and database["foreign_key_violations"] == 0
            and counts["mismatched"] == 0
            and counts["missing"] == 0
        )
        manifest = {
            "format": BACKUP_FORMAT,
            "label": label,
            "created_at": created_at,
            "sqlite_version": sqlite3.sqlite_version,
            "source": {"database_name": db_path.name, "paths": "relative; absolute source paths are not recorded"},
            "database": database,
            "artifacts": [
                {
                    "rel_path": e.rel_path,
                    "status": e.status,
                    "sha256": e.sha256,
                    "size_bytes": e.size_bytes,
                    "cataloged_sha256": e.cataloged_sha256,
                }
                for e in entries
            ],
            "summary": {**counts, "cataloged": len(catalog)},
            "vanished_during_backup": sorted(vanished),
            "complete": complete,
            "durability_note": "files fsynced; power-loss durability not verified",
        }
        body = canonical_json(manifest).encode("utf-8")
        digest = sha256_hex(body)
        with open(staging / MANIFEST_FILENAME, "xb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        with open(staging / MANIFEST_SHA_FILENAME, "x", encoding="ascii", newline="\n") as fh:
            fh.write(f"{digest}  {MANIFEST_FILENAME}\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(staging, dest_dir)  # fails instead of replacing if dest appeared meanwhile
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # only our own partial copy
        raise
    return BackupResult(path=dest_dir, manifest=manifest, manifest_sha256=digest, artifacts=tuple(entries))


# -- verification --------------------------------------------------------------


def load_manifest(backup_dir: Path) -> tuple[dict[str, Any], str]:
    """Manifest and its hash, refusing a manifest whose recorded SHA-256 differs."""
    backup_dir = Path(backup_dir)
    body = (backup_dir / MANIFEST_FILENAME).read_bytes()
    recorded = (backup_dir / MANIFEST_SHA_FILENAME).read_text(encoding="ascii").split()[0]
    digest = sha256_hex(body)
    if digest != recorded:
        raise BackupError(f"manifest hash {digest} differs from recorded {recorded}")
    manifest = json.loads(body)
    if manifest.get("format") != BACKUP_FORMAT:
        raise BackupError(f"unsupported backup format {manifest.get('format')!r}")
    return manifest, digest


def verify_backup(backup_dir: Path) -> BackupVerification:
    """Re-hash everything in a backup against its manifest; reads only."""
    backup_dir = Path(backup_dir)
    problems: list[str] = []
    try:
        manifest, _ = load_manifest(backup_dir)
    except (OSError, ValueError, BackupError) as exc:
        return BackupVerification(backup_dir, (f"manifest: {exc}",))
    db_file = backup_dir / DB_FILENAME
    database = manifest["database"]
    if not db_file.is_file():
        problems.append("database copy is missing")
    else:
        sha, size = _sha256_file(db_file)
        if (sha, size) != (database["sha256"], database["size_bytes"]):
            problems.append("database copy hash/size differs from manifest")
        conn = sqlite3.connect(_ro_uri(db_file, immutable=True), uri=True)
        try:
            if [r[0] for r in conn.execute("PRAGMA integrity_check")] != ["ok"]:
                problems.append("database copy fails integrity_check")
            if table_digests(conn) != database["tables"]:
                problems.append("database copy table digests differ from manifest")
        except sqlite3.DatabaseError as exc:
            problems.append(f"database copy is unreadable: {exc}")
        finally:
            conn.close()
    expected_files: set[str] = set()
    for entry in manifest["artifacts"]:
        if entry["status"] in ("missing", "temporary"):
            continue
        expected_files.add(entry["rel_path"])
        path = backup_dir / ARTIFACTS_DIRNAME / PurePosixPath(entry["rel_path"])
        if not path.is_file():
            problems.append(f"artifact copy missing: {entry['rel_path']}")
            continue
        if _sha256_file(path) != (entry["sha256"], entry["size_bytes"]):
            problems.append(f"artifact copy hash differs: {entry['rel_path']}")
    extra = {rel for rel, _ in _walk_files(backup_dir / ARTIFACTS_DIRNAME)} - expected_files
    problems.extend(f"unexpected file in backup: {rel}" for rel in sorted(extra))
    return BackupVerification(backup_dir, tuple(problems))


# -- restore -------------------------------------------------------------------


def restore_backup(backup_dir: Path, target_db: Path, target_artifact_root: Path) -> RestoreReport:
    """Restore into new, empty targets and check the result against the manifest.

    Refuses to write over an existing database (or its -wal/-shm) or into a
    non-empty artifact root: restoring never replaces live evidence.
    """
    backup_dir, target_db, target_artifact_root = Path(backup_dir), Path(target_db), Path(target_artifact_root)
    verification = verify_backup(backup_dir)
    if not verification.ok:
        raise BackupError(f"backup does not verify: {list(verification.problems)[:5]}")
    for candidate in (target_db, Path(str(target_db) + "-wal"), Path(str(target_db) + "-shm")):
        if candidate.exists():
            raise BackupError(f"restore target exists, refusing to overwrite: {candidate}")
    if target_artifact_root.exists() and any(target_artifact_root.iterdir()):
        raise BackupError(f"restore artifact root is not empty: {target_artifact_root}")
    if _is_within(target_db, target_artifact_root):
        raise BackupError("the restored database must live outside the artifact root")
    manifest, _ = load_manifest(backup_dir)

    target_db.parent.mkdir(parents=True, exist_ok=True)
    _copy_hashing(backup_dir / DB_FILENAME, target_db)
    target_artifact_root.mkdir(parents=True, exist_ok=True)
    for rel, full in _walk_files(backup_dir / ARTIFACTS_DIRNAME):
        _copy_hashing(full, target_artifact_root / PurePosixPath(rel))

    with SQLiteRunStore(target_db, target_artifact_root) as store:  # verifies migration hashes
        report = store.check_integrity(hash_files=True)
        restored_tables = table_digests(store._conn)
        catalog = store.artifact_catalog()
    expected_tables = manifest["database"]["tables"]
    mismatches = sorted(name for name, info in expected_tables.items() if restored_tables.get(name) != info)
    expected_catalog = {
        e["rel_path"]: e["cataloged_sha256"]
        for e in manifest["artifacts"]
        if e["status"] in ("verified", "mismatched", "missing")
    }
    invariants = [
        *(f"valid attempt without measurement: {a}" for a in report.valid_without_measurement),
        *(f"ledger mismatch: {a}" for a in report.ledger_mismatches),
    ]
    return RestoreReport(
        db_path=target_db,
        artifact_root=target_artifact_root,
        backup_complete=bool(manifest["complete"]),
        integrity_messages=report.sqlite_messages,
        foreign_key_violations=len(report.foreign_key_violations),
        invariant_problems=tuple(invariants),
        table_mismatches=tuple(mismatches),
        catalog_matches={k.casefold(): v for k, v in catalog.items()}
        == {k.casefold(): v for k, v in expected_catalog.items()},
        missing=report.artifacts.missing,
        mismatched=report.artifacts.mismatched,
        orphans=report.artifacts.orphans,
        temporaries=report.artifacts.temporaries,
    )


def rehearse_restore(backup_dir: Path, workdir: Path) -> RestoreReport:
    """Restore into a fresh scratch directory (never the live paths) and report."""
    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise BackupError(f"rehearsal directory must be empty: {workdir}")
    return restore_backup(backup_dir, workdir / "db" / DB_FILENAME, workdir / ARTIFACTS_DIRNAME)
