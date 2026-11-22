"""Exact code version of an execution: operator label + git SHA + dirty state.

Nothing is invented: when git is unavailable or the tree cannot be read, the
fields stay `None` and `error` says why. `dirty` covers tracked files
(staged or not); untracked files under `src/` are code that would run too,
so their names and contents enter the fingerprint as well. Other untracked
files (editor or tool folders) are only counted.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cnnlab

CODE_VERSION_FORMAT = "tcc-code-version/1"
_GIT_TIMEOUT_S = 30


@dataclass(frozen=True, slots=True)
class CodeVersion:
    label: str
    git_sha: str | None
    dirty: bool | None
    dirty_fingerprint: str | None
    modified_paths: tuple[str, ...]
    untracked_code_paths: tuple[str, ...]
    untracked_other_count: int
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CODE_VERSION_FORMAT,
            "label": self.label,
            "git_sha": self.git_sha,
            "dirty": self.dirty,
            "dirty_fingerprint": self.dirty_fingerprint,
            "modified_paths": list(self.modified_paths),
            "untracked_code_paths": list(self.untracked_code_paths),
            "untracked_other_count": self.untracked_other_count,
            "error": self.error,
        }

    @property
    def identity(self) -> tuple[str | None, str | None]:
        """What must match to call two sessions the same code (the label is only a name)."""
        return (self.git_sha, self.dirty_fingerprint)

    @property
    def is_known(self) -> bool:
        return self.git_sha is not None and self.dirty_fingerprint is not None


def code_identity(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return (payload.get("git_sha"), payload.get("dirty_fingerprint"))


def default_repo_root() -> Path:
    """Repository holding the `cnnlab` package that is actually imported."""
    return Path(cnnlab.__file__).resolve().parents[2]


def require_imported_code(repo_root: Path | None) -> Path:
    """The repository root, after checking that it is the one the running ``cnnlab`` was imported from.

    ``--repo-root`` tells where to read the commit and the dirty state *from*; if the code that actually runs came
    from elsewhere (the editable install of another checkout), the record would describe code that did not run.
    To run from a fixed worktree, put ``<worktree>/src`` first in ``PYTHONPATH``; the children inherit it.
    """
    imported = default_repo_root()
    if repo_root is None:
        return imported
    given = Path(repo_root).resolve()
    if given != imported:
        raise ValueError(
            f"--repo-root is {given} but the running cnnlab was imported from {imported}: the commit that would be "
            f"recorded is not the code that runs. Put {given / 'src'} first in PYTHONPATH (and run again)"
        )
    return imported


def tags_at_head(repo: Path) -> tuple[str, ...]:
    """Git tags that point at HEAD of ``repo`` (sorted); empty when there are none or git cannot say."""
    try:
        out = _git(Path(repo), "tag", "--points-at", "HEAD").decode().split()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return ()
    return tuple(sorted(out))


def _git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def capture_code_version(label: str, repo_root: Path | None = None) -> CodeVersion:
    """Read HEAD, tracked changes and untracked code of `repo_root` (default: imported code)."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("a code version needs an explicit, non-empty label")
    repo = Path(repo_root) if repo_root is not None else default_repo_root()
    try:
        sha = _git(repo, "rev-parse", "HEAD").decode().strip()
        status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        diff = _git(repo, "diff", "HEAD", "--binary")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return CodeVersion(label, None, None, None, (), (), 0, error=f"{type(exc).__name__}: {exc}")

    modified: list[str] = []
    untracked_code: list[str] = []
    untracked_other = 0
    entries = [e for e in status.decode("utf-8", errors="replace").split("\0") if e]
    skip_next = False
    for entry in entries:
        if skip_next:  # second path of a rename/copy entry
            skip_next = False
            continue
        code, path = entry[:2], entry[3:]
        if code[0] in "RC":
            skip_next = True
        if code == "??":
            if path.startswith("src/"):
                untracked_code.append(path)
            else:
                untracked_other += 1
        else:
            modified.append(path)

    digest = hashlib.sha256()
    digest.update(sha.encode())
    digest.update(b"\0diff\0")
    digest.update(diff)
    for path in sorted(untracked_code):
        digest.update(b"\0untracked\0" + path.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((repo / path).read_bytes()).digest())
    return CodeVersion(
        label=label,
        git_sha=sha,
        dirty=bool(modified or untracked_code),
        dirty_fingerprint=digest.hexdigest(),
        modified_paths=tuple(sorted(modified)),
        untracked_code_paths=tuple(sorted(untracked_code)),
        untracked_other_count=untracked_other,
    )
