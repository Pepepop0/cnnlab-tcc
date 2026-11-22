"""Code version capture: git SHA + dirty state, never invented (temporary git repos only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cnnlab.tcc.operation.provenance import capture_code_version, code_identity


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_tree_reports_sha_and_not_dirty(repo: Path) -> None:
    cv = capture_code_version("demo-label", repo)
    assert cv.is_known and cv.dirty is False and cv.error is None
    assert len(cv.git_sha) == 40 and cv.modified_paths == () and cv.untracked_code_paths == ()
    assert cv.to_payload()["label"] == "demo-label"
    assert code_identity(cv.to_payload()) == cv.identity


def test_tracked_and_untracked_code_change_the_fingerprint(repo: Path) -> None:
    clean = capture_code_version("x", repo)
    (repo / "src" / "pkg" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    edited = capture_code_version("x", repo)
    assert edited.dirty and edited.modified_paths == ("src/pkg/mod.py",)
    assert edited.git_sha == clean.git_sha and edited.dirty_fingerprint != clean.dirty_fingerprint

    (repo / "src" / "pkg" / "new.py").write_text("Y = 1\n", encoding="utf-8")
    with_new = capture_code_version("x", repo)
    assert with_new.untracked_code_paths == ("src/pkg/new.py",)
    assert with_new.dirty_fingerprint != edited.dirty_fingerprint
    (repo / "src" / "pkg" / "new.py").write_text("Y = 2\n", encoding="utf-8")
    assert capture_code_version("x", repo).dirty_fingerprint != with_new.dirty_fingerprint  # content counts

    (repo / "notes.txt").write_text("not code", encoding="utf-8")
    other = capture_code_version("x", repo)
    assert (
        other.untracked_other_count == 1
        and other.dirty_fingerprint == capture_code_version("x", repo).dirty_fingerprint
    )


def test_label_is_only_a_name(repo: Path) -> None:
    assert capture_code_version("a", repo).identity == capture_code_version("b", repo).identity
    with pytest.raises(ValueError):
        capture_code_version("  ", repo)


def test_outside_git_nothing_is_invented(tmp_path: Path) -> None:
    cv = capture_code_version("x", tmp_path / "not-a-repo")
    assert cv.git_sha is None and cv.dirty is None and cv.dirty_fingerprint is None
    assert not cv.is_known and cv.error
