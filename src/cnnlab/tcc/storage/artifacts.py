"""Artifact files: safe relative paths, atomic publication and catalog scan.

Large files never go into SQLite. The order required by TCC-04/TCC-09 is:
write to a temporary file in the destination directory, `fsync`, hash, then
`os.replace` into place; only after that the catalog row is committed. A crash
between the two steps leaves an *orphan* file, which `scan_artifacts` reports;
a catalog row whose file vanished is reported as *missing*.

Catalog paths are POSIX, relative to a dedicated artifact root, restricted to
`[A-Za-z0-9._-]` components, so the same export works on Windows and Linux and
no path can escape the root. Temporaries start with `.` and therefore can never
be cataloged by mistake.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cnnlab.tcc.storage.errors import ArtifactIntegrityError

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)
TEMP_SUFFIX = ".tmp"
_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    rel_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactScan:
    """Differences between catalog and disk. Empty tuples mean consistent."""

    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    orphans: tuple[str, ...]
    temporaries: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.mismatched or self.orphans or self.temporaries)


def _check_component(part: str) -> None:
    if not _SAFE_COMPONENT.match(part):
        raise ArtifactIntegrityError(f"unsafe path component {part!r}")
    if part.split(".", 1)[0].lower() in _WINDOWS_RESERVED:
        raise ArtifactIntegrityError(f"reserved Windows name {part!r}")


def normalize_rel_path(path: str) -> str:
    """Validate a catalog path and return it unchanged; reject instead of fixing."""
    if not isinstance(path, str) or not path:
        raise ArtifactIntegrityError("artifact path must be a non-empty string")
    if "\\" in path or ":" in path or path.startswith("/"):
        raise ArtifactIntegrityError(f"artifact path must be relative POSIX: {path!r}")
    for part in path.split("/"):
        _check_component(part)
    return path


def safe_component(value: str) -> str:
    """Validate an identifier used as a directory name (run_id, attempt_id, kind)."""
    _check_component(value)
    return value


def attempt_artifact_dir(run_id: str, attempt_id: str) -> str:
    """Recommended layout `runs/<run_id>/attempts/<attempt_id>` (relative to the root)."""
    return str(PurePosixPath("runs", safe_component(run_id), "attempts", safe_component(attempt_id)))


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(directory: Path) -> None:
    # Directory fsync makes the rename durable on POSIX; Windows cannot open dirs.
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _destination(root: Path, rel_path: str, overwrite: bool) -> Path:
    dest = Path(root) / PurePosixPath(normalize_rel_path(rel_path))
    if dest.exists() and not overwrite:
        raise ArtifactIntegrityError(f"artifact already exists: {rel_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _temp_path(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.{secrets.token_hex(6)}{TEMP_SUFFIX}")


def _publish(tmp: Path, dest: Path, rel_path: str) -> PublishedArtifact:
    sha, size = sha256_file(tmp)
    os.replace(tmp, dest)
    _fsync_dir(dest.parent)
    return PublishedArtifact(rel_path=rel_path, sha256=sha, size_bytes=size)


def publish_bytes(root: Path, rel_path: str, data: bytes, *, overwrite: bool = False) -> PublishedArtifact:
    """Write `data` atomically under `root`; the returned hash goes into `EvaluationResult`."""
    dest = _destination(root, rel_path, overwrite)
    tmp = _temp_path(dest)
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        return _publish(tmp, dest, rel_path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_file(root: Path, rel_path: str, source: Path, *, overwrite: bool = False) -> PublishedArtifact:
    """Copy `source` into the catalog atomically; the source is left untouched."""
    dest = _destination(root, rel_path, overwrite)
    tmp = _temp_path(dest)
    try:
        with open(source, "rb") as src, open(tmp, "xb") as fh:
            shutil.copyfileobj(src, fh, _CHUNK)
            fh.flush()
            os.fsync(fh.fileno())
        return _publish(tmp, dest, rel_path)
    finally:
        tmp.unlink(missing_ok=True)


def verify_file(root: Path, rel_path: str, expected_sha256: str) -> int:
    """Check presence and hash of a cataloged file; return its size in bytes."""
    path = Path(root) / PurePosixPath(normalize_rel_path(rel_path))
    if not path.is_file():
        raise ArtifactIntegrityError(f"artifact file is missing: {rel_path}")
    sha, size = sha256_file(path)
    if sha != expected_sha256:
        raise ArtifactIntegrityError(f"artifact hash mismatch for {rel_path}: expected {expected_sha256}, found {sha}")
    return size


def scan_artifacts(root: Path, catalog: Mapping[str, str], *, hash_files: bool = True) -> ArtifactScan:
    """Compare `catalog` (rel_path -> sha256) with every file below `root`.

    The root must be dedicated to artifacts: any uncataloged file is an orphan.
    Comparison is case-insensitive because the catalog is (Windows semantics).
    """
    root = Path(root)
    by_key = {path.casefold(): path for path in catalog}
    seen: set[str] = set()
    orphans: list[str] = []
    temporaries: list[str] = []
    mismatched: list[str] = []
    if root.exists():
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                full = Path(dirpath) / name
                rel = full.relative_to(root).as_posix()
                if name.startswith(".") and name.endswith(TEMP_SUFFIX):
                    temporaries.append(rel)
                    continue
                key = rel.casefold()
                if key not in by_key:
                    orphans.append(rel)
                    continue
                seen.add(key)
                cataloged = by_key[key]
                if hash_files and sha256_file(full)[0] != catalog[cataloged]:
                    mismatched.append(cataloged)
    missing = [path for key, path in by_key.items() if key not in seen]
    return ArtifactScan(
        missing=tuple(sorted(missing)),
        mismatched=tuple(sorted(mismatched)),
        orphans=tuple(sorted(orphans)),
        temporaries=tuple(sorted(temporaries)),
    )
