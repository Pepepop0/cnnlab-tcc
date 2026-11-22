"""The campaign runs from a fixed worktree on the frozen commit while the main checkout moves on. CPU only.

A real second worktree of this repository is made at HEAD, given one more commit (so its SHA is not the checkout's
under test) and a tag. Everything the "Passo a passo da campanha" does runs in subprocesses with ``PYTHONPATH`` pointing
at that worktree's ``src``: the plan is frozen from ``piloto.json``, the runs are queued, the executor runs them with
the synthetic wiring. The provenance in the database must be the worktree's commit, clean, and a command whose
``--repo-root`` is not the code it runs must refuse.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from cnnlab.tcc.operation.demo import EXIT_REFUSED, default_repo_root
from cnnlab.tcc.storage import ReadModel

REPO = default_repo_root()
FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
REFUSED = EXIT_REFUSED


def git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return done.stdout.strip()


@pytest.fixture(scope="module")
def worktree(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, str, str]]:
    if shutil.which("git") is None:
        pytest.skip("git is needed to make a worktree")
    base = tmp_path_factory.mktemp("campanha")
    wt, tag = base / "worktree", f"teste-campanha-{uuid.uuid4().hex[:8]}"
    git("-C", str(REPO), "worktree", "add", "--detach", str(wt), "HEAD")
    try:
        git("commit", "--allow-empty", "-m", "freeze of the test campaign", cwd=wt)
        git("tag", tag, cwd=wt)
        yield wt, git("rev-parse", "HEAD", cwd=wt), tag
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)], capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "tag", "-d", tag], capture_output=True)


def cli(wt: Path, *args: str, src: Path | None = None, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    """``python -m cnnlab.tcc.operation.cli ...`` run from ``wt`` with ``PYTHONPATH`` on ``src`` (default: wt/src)."""
    env = {
        **os.environ,
        "PYTHONPATH": str(src if src is not None else wt / "src"),
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [sys.executable, "-m", "cnnlab.tcc.operation.cli", *args],
        cwd=wt,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def files(tmp: Path) -> dict[str, Path]:
    settings = tmp / "S.json"
    settings.write_text(
        json.dumps(
            {
                "master_seed": 20261026,
                "factory": FACTORY,
                "methods": ["joint"],
                "factory_args": {"slice": "admitted", "n_candidates": 3, "sleep_s": 0.0},
                "timing_cache": {"adopted": False},
            }
        ),
        encoding="utf-8",
    )
    pilot = tmp / "piloto.json"
    pilot.write_text(
        json.dumps(
            {
                "run_id": "piloto-aleatorio",
                "viable_minima": {"E": 0.2, "L_ms": 0.5, "V_mb": 2.0, "S_mb": 1.0},
                "decisions": {
                    "search_epochs": 5,
                    "latency_repetitions": 1000,
                    "train_time_limit_s": 1800,
                    "reference_point": {"E": 1.0, "L_ms": 2.5, "V_mb": 12.5, "S_mb": 5.0},
                },
            }
        ),
        encoding="utf-8",
    )
    queue = tmp / "fila"
    return {
        "settings": settings,
        "pilot": pilot,
        "queue": queue,
        "db": queue / "db" / "tcc.sqlite3",
        "art": queue / "artifacts",
    }


def test_where_names_the_code_that_runs_its_commit_and_its_tag(worktree: tuple[Path, str, str]) -> None:
    wt, sha, tag = worktree
    done = cli(wt, "campaign", "where", "--repo-root", str(wt))
    assert done.returncode == 0, done.stderr
    assert f"cnnlab imported from {(wt / 'src' / 'cnnlab').resolve()}" in done.stdout
    assert f"commit {sha}" in done.stdout and tag in done.stdout and "uncommitted changes no" in done.stdout


def test_a_repo_root_that_is_not_the_code_that_runs_is_refused_everywhere(
    worktree: tuple[Path, str, str], tmp_path: Path
) -> None:
    wt, _, _ = worktree
    f = files(tmp_path)
    elsewhere = REPO / "src"  # the checkout under test: not the worktree
    where = cli(wt, "campaign", "where", "--repo-root", str(wt), src=elsewhere)
    assert where.returncode == 2 and "NOT the root of the running code" in where.stdout
    for args in (
        ("campaign", "plan", "--settings", str(f["settings"]), "--repo-root", str(wt), "--allow-synthetic"),
        (
            "campaign",
            "enqueue",
            "--db",
            str(f["db"]),
            "--plan",
            "x",
            "--queue-dir",
            str(f["queue"]),
            "--repo-root",
            str(wt),
        ),
        ("fila", "--dir", str(f["queue"]), "--repo-root", str(wt)),
    ):
        done = cli(wt, *args, src=elsewhere)
        assert done.returncode == REFUSED and "was imported from" in done.stderr and "PYTHONPATH" in done.stderr, args
    assert not f["queue"].exists()


def test_the_whole_campaign_runs_from_the_worktree_and_records_its_commit(
    worktree: tuple[Path, str, str], tmp_path: Path
) -> None:
    wt, sha, tag = worktree
    f = files(tmp_path)
    plan = cli(
        wt,
        "campaign",
        "plan",
        "--settings",
        str(f["settings"]),
        "--pilot",
        str(f["pilot"]),
        "--db",
        str(f["db"]),
        "--artifacts",
        str(f["art"]),
        "--freeze",
        "--repo-root",
        str(wt),
        "--allow-synthetic",
    )
    assert plan.returncode == 0, plan.stdout + plan.stderr
    plan_id = next(w for w in plan.stdout.split() if w.startswith("campaign-plan-"))
    where = cli(wt, "campaign", "where", "--repo-root", str(wt), "--db", str(f["db"]), "--plan", plan_id)
    assert where.returncode == 0 and "this checkout matches" in where.stdout and tag in where.stdout, where.stdout
    with ReadModel(f["db"]) as rm:
        frozen = rm.plan(plan_id)["content"]["code_freeze"]
    assert frozen["git_sha"] == sha and frozen["tags"] == [tag] and frozen["dirty"] is False

    queued = cli(
        wt,
        "campaign",
        "enqueue",
        "--db",
        str(f["db"]),
        "--plan",
        plan_id,
        "--queue-dir",
        str(f["queue"]),
        "--repo-root",
        str(wt),
    )
    assert queued.returncode == 0 and "queued 8 of 8 runs" in queued.stdout, queued.stdout + queued.stderr

    ran = cli(wt, "fila", "--dir", str(f["queue"]), "--repo-root", str(wt), "--allow-synthetic", "--poll-s", "0.2")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    with ReadModel(f["db"]) as rm:
        runs = rm.runs()
        jobs = rm.jobs()
    assert (
        len(runs) == 8 and {r["stage"] for r in runs} == {"confirmatory"} and {r["plan_id"] for r in runs} == {plan_id}
    )
    assert {j["state"] for j in jobs} == {"done"}
    # the provenance is the worktree's commit, clean: not the main checkout's
    assert {r["git_sha"] for r in runs} == {sha} and {r["code_dirty"] for r in runs} == {False}
    assert sha != git("rev-parse", "HEAD", cwd=REPO)


def test_a_worktree_moved_off_the_frozen_commit_cannot_queue_more(
    worktree: tuple[Path, str, str], tmp_path: Path
) -> None:
    wt, sha, _ = worktree
    f = files(tmp_path)
    plan = cli(
        wt,
        "campaign",
        "plan",
        "--settings",
        str(f["settings"]),
        "--pilot",
        str(f["pilot"]),
        "--db",
        str(f["db"]),
        "--artifacts",
        str(f["art"]),
        "--freeze",
        "--repo-root",
        str(wt),
        "--allow-synthetic",
    )
    assert plan.returncode == 0, plan.stdout + plan.stderr
    plan_id = next(w for w in plan.stdout.split() if w.startswith("campaign-plan-"))
    stray = wt / "src" / "cnnlab" / "tcc" / "stray_edit.py"
    stray.write_text("X = 1\n", encoding="utf-8")  # code that would run and that no commit holds
    try:
        refused = cli(
            wt,
            "campaign",
            "enqueue",
            "--db",
            str(f["db"]),
            "--plan",
            plan_id,
            "--queue-dir",
            str(f["queue"]),
            "--repo-root",
            str(wt),
        )
        assert refused.returncode == REFUSED and "uncommitted changes" in refused.stderr
        where = cli(wt, "campaign", "where", "--repo-root", str(wt), "--db", str(f["db"]), "--plan", plan_id)
        assert where.returncode == 2 and "uncommitted changes yes" in where.stdout and "DOES NOT match" in where.stdout
    finally:
        stray.unlink()
