"""`cnnlab-tcc` entry point: thin shell over the operation layer (CPU, synthetic wiring opted in)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import cli, launcher
from cnnlab.tcc.operation.demo import EXIT_REFUSED, DemoRunSpec
from cnnlab.tcc.storage import ReadModel, create_backup
from cnnlab.tcc.storage.testing import make_v1_copy

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
NO_GPU = {"CUDA_VISIBLE_DEVICES": "-1"}


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cnnlab.tcc.operation.cli", *args],
        env=launcher.cli_environment(NO_GPU),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    spec = tmp_path / "spec.json"
    payload: dict[str, Any] = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 11,
        "train_seed": 22,
        "calibration_seed": 33,
        "budget_gpu_s": 3600.0,
        "code_label": "cli-test",
        "author": "test-suite",
        "factory": FACTORY,
        "factory_args": {"n_candidates": 3, "sleep_s": 0.0},
    }
    spec.write_text(json.dumps(DemoRunSpec.from_mapping(payload).to_payload()), encoding="utf-8")
    return tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts", spec


def test_run_status_verify_backup_and_recorded_provenance(paths: tuple[Path, Path, Path], tmp_path: Path) -> None:
    db, art, spec = paths
    base = ["--db", str(db), "--artifacts", str(art)]
    refused = _run(["run", *base, "--spec", str(spec)])  # synthetic wiring is never the default
    assert refused.returncode == EXIT_REFUSED and "SyntheticWiringError" in refused.stderr and not db.exists()

    done = _run(["run", *base, "--spec", str(spec), "--allow-synthetic", "--worker-id", "cli"])
    assert done.returncode == 0, done.stderr
    assert "ended done" in done.stdout

    status = _run(["status", "--db", str(db)])
    assert status.returncode == 0 and "[done]" in status.stdout and "SYNTHETIC evaluator" in status.stdout
    with ReadModel(db) as rm:
        [run] = rm.runs()
    # Provenance of every run: readable version from pyproject, git SHA and the dirty flag.
    assert run["version_source"] == "operation-manifest" and run["code_label"] == "cli-test"
    assert run["system_version"].startswith("cnnlab-tcc 0.1.0 | ") and "pre-release" in run["system_version"]
    assert len(run["git_sha"]) == 40 and isinstance(run["code_dirty"], bool)
    assert ("+dirty" in run["system_version"]) == run["code_dirty"]

    verified = _run(["verify", *base])
    assert verified.returncode == 0, verified.stdout
    assert verified.stdout.strip().endswith("3 cataloged, 3 verified, 0 missing, 0 mismatched, 0 orphan, 0 temporary")

    backup = _run(["backup", *base, "--dest", str(tmp_path / "bk"), "--label", "cli-test"])
    assert backup.returncode == 0 and "complete=True" in backup.stdout
    assert _run(["verify", "--backup", str(tmp_path / "bk")]).returncode == 0

    next(art.rglob("engine.plan")).write_bytes(b"changed after commit")
    damaged = _run(["verify", *base])
    assert damaged.returncode == cli.EXIT_PROBLEMS and "mismatched: runs/" in damaged.stdout


def test_stop_is_a_cooperative_cancel_and_never_creates_or_migrates(
    paths: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    db, art, spec = paths
    base = ["--db", str(db), "--artifacts", str(art)]
    missing = _run(["stop", *base, "--job-id", "j", "--reason", "x"])
    assert missing.returncode == EXIT_REFUSED and "ReadModelError" in missing.stderr and not db.exists()

    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    with ReadModel(db) as rm:
        job_id = rm.jobs()[0]["job_id"]
    stopped = _run(["stop", *base, "--job-id", job_id, "--reason", "ensaio"])
    assert stopped.returncode == 0 and f"job {job_id}: cancelled" in stopped.stdout

    old = make_v1_copy(db, tmp_path / "v1" / "demo.sqlite")
    before = old.read_bytes()
    refused = _run(["stop", "--db", str(old), "--artifacts", str(art), "--job-id", job_id, "--reason", "x"])
    assert refused.returncode == EXIT_REFUSED and "MigrationNeededError" in refused.stderr
    assert _run(["verify", "--db", str(old), "--artifacts", str(art)]).stdout.startswith("schema v1: ")
    assert old.read_bytes() == before


def test_help_unknown_command_and_lazy_interfaces(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["--help"]) == 0 and "cnnlab-tcc run" in capsys.readouterr().out
    assert cli.main([]) == cli.EXIT_PROBLEMS
    assert cli.main(["explode"]) == cli.EXIT_PROBLEMS and "unknown command" in capsys.readouterr().err
    assert "epito" not in cli.__doc__.replace("cnnlab", "")  # no new text with the old project name

    import cnnlab.tcc.demo_streamlit as page
    import cnnlab.tcc.demo_tui as tui

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(tui, "main", lambda argv: seen.append(("tui", list(argv))) or 0)
    monkeypatch.setattr(page, "main", lambda argv: seen.append(("view", list(argv))) or 0)
    assert cli.main(["tui", "--db", "a", "--artifacts", "b", "--read-only"]) == 0
    assert cli.main(["view", "--db", "a", "--server.port", "8502"]) == 0
    assert seen == [
        ("tui", ["--db", "a", "--artifacts", "b", "--read-only"]),
        ("view", ["--db", "a", "--server.port", "8502"]),
    ]
    assert os.environ.get("TCC_CLI_DEFAULT_DB") is None  # paths are always explicit
