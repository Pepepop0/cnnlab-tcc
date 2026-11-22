"""`cnnlab-tcc export`: CSV files and manifest of one run, read-only (CPU only).

The stored runs use the synthetic test wiring, opted into explicitly; their
numbers are stand-ins and leave only in columns labeled `_SINTETICO`. Rows
used to check real columns are in-memory dicts in the read-model shape,
never stored as results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import cli, export
from cnnlab.tcc.operation.demo import EXIT_REFUSED, DemoRunSpec, start_demo
from cnnlab.tcc.operation.provenance import CodeVersion
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import ReadModel
from cnnlab.tcc.storage.errors import NotFoundError
from cnnlab.tcc.storage.testing import make_v1_copy

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
CODE = CodeVersion("export-test", "d" * 40, False, "4" * 64, (), (), 0)
FILES = ("medidas.csv", "tentativas.csv", "geracoes.csv", "frentes.csv", "checkpoints.csv", "tempo_gpu.csv")


@pytest.fixture
def run_db(tmp_path: Path) -> tuple[Path, str]:
    db, art = tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"
    spec = DemoRunSpec.from_mapping(
        {
            "method": "joint",
            "seed_block": 0,
            "search_seed": 11,
            "train_seed": 22,
            "calibration_seed": 33,
            "budget_gpu_s": 3600.0,
            "code_label": "export-test",
            "author": "test-suite",
            "factory": FACTORY,
            "factory_args": {"n_candidates": 3, "sleep_s": 0.0},
        }
    )
    result = start_demo(db, art, spec, synthetic_demo_wiring, worker_id="w", code_version=CODE, allow_synthetic=True)
    assert result.outcome is not None and result.outcome.end_state == "done"
    return db, result.run_id


def _table(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_run_is_refused_by_default_and_labeled_with_opt_in(run_db: tuple[Path, str], tmp_path: Path) -> None:
    db, run_id = run_db
    before = _digest(db)
    dest = tmp_path / "out"
    with pytest.raises(export.ExportError, match="synthetic result"):
        export.export_run(db, run_id, dest)
    assert not dest.exists() and list(tmp_path.glob(".out.partial-*")) == []  # refused: nothing written

    result = export.export_run(db, run_id, dest, allow_synthetic=True)
    assert result.path == dest and sorted(p.name for p in dest.iterdir()) == sorted((*FILES, export.MANIFEST_NAME))
    assert _digest(db) == before and export.verify_export(dest) == []
    manifest = json.loads((dest / export.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "tcc-export/1" and manifest["read_model"] == "tcc-read/1"
    assert manifest["synthetic"] is True and "_SINTETICO" in manifest["synthetic_note"]
    assert manifest["confirmatory"] is False and manifest["changed_during_export"] is False
    with ReadModel(db) as rm:
        [run] = rm.runs()
    assert manifest["run"]["run_id"] == run_id and manifest["run"]["space_hash"] == run["space_hash"]
    assert manifest["run"]["git_sha"] == "d" * 40 and manifest["run"]["code_dirty"] is False
    assert manifest["run"]["plan_sha256"] == run["plan_sha256"] and manifest["attempts_by_status"] == {"valid": 3}
    assert [f["name"] for f in manifest["files"]] == list(FILES)
    for entry in manifest["files"]:
        data = (dest / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"]
        assert data.decode("utf-8").splitlines()[0].split(",") == entry["columns"] and b"\r" not in data

    measures = _table(dest / "medidas.csv")
    assert [m["attempt_id"] for m in measures] == ["c0.a1", "c1.a1", "c2.a1"]
    numbers = ("E_erro_top1", "L_ms", "V_MB", "S_MB", "gpu_s")
    assert all(f"{n}_SINTETICO" in measures[0] and n not in measures[0] for n in numbers)  # no unlabeled number
    assert {m["sintetico"] for m in measures} == {"true"}
    with ReadModel(db) as rm:
        stored = rm.evaluations(run_id)
    assert [float(m["E_erro_top1_SINTETICO"]) for m in measures] == [r["e"] for r in stored]  # full value, not rounded
    assert [float(m["L_ms_SINTETICO"]) for m in measures] == [r["l_ms"] for r in stored]

    attempts = _table(dest / "tentativas.csv")
    assert len(attempts) == 3 and "gpu_s_SINTETICO" in attempts[0] and "gpu_s" not in attempts[0]
    assert {a["status"] for a in attempts} == {"valid"} and {a["failure_code"] for a in attempts} == {""}
    [generation] = _table(dest / "geracoes.csv")
    assert generation["validas"] == "3" and "menor_E_erro_top1_SINTETICO" in generation
    gpu = _table(dest / "tempo_gpu.csv")
    assert [g["escopo"] for g in gpu] == ["run (livro de GPU)", "tentativas valid"]
    assert float(gpu[0]["gpu_ativa_s_SINTETICO"]) == 30.0 and gpu[0]["orcamento_gpu_s"] == "3600.0"
    assert len(_table(dest / "checkpoints.csv")) >= 1 and _table(dest / "frentes.csv") == []

    with pytest.raises(export.ExportError, match="already exists"):
        export.export_run(db, run_id, dest, allow_synthetic=True)  # never overwrites
    with pytest.raises(NotFoundError):
        export.export_run(db, "no-such-run", tmp_path / "other", allow_synthetic=True)
    with pytest.raises(export.ExportError, match="parent of the destination"):
        export.export_run(db, run_id, tmp_path / "missing" / "out", allow_synthetic=True)
    (dest / "medidas.csv").write_bytes(b"changed")
    (dest / "frentes.csv").unlink()
    assert export.verify_export(dest) == ["mismatched: medidas.csv", "missing: frentes.csv"]


def _row(i: int, **kw: Any) -> dict[str, Any]:
    """In-memory row in the read-model shape (formatting input only; never stored)."""
    row = {
        "attempt_id": f"c{i}.a1",
        "candidate_id": f"c{i}",
        "generation": 0,
        "attempt_number": 1,
        "status": "valid",
        "cache_hit": False,
        "source_attempt_id": None,
        "e": 0.123456789012,
        "l_ms": 1.25,
        "v_mb": 12.5,
        "s_mb": 3.75,
        "gpu_seconds": 61.0,
        "failure_stage": None,
        "failure_code": None,
        "recovered": False,
        "synthetic": False,
        "preliminary": True,
        "not_confirmatory": True,
        "phenotype_hash": "p",
        "cache_context_hash": "c",
        "started_at": "2026-09-29T10:00:00.000000Z",
        "committed_at": "2026-09-29T10:01:01.000000Z",
        "commit_seq": i + 1,
    }
    row.update(kw)
    return row


def test_real_rows_use_plain_columns_and_unmeasured_stays_empty() -> None:
    rows = [
        _row(0),
        _row(
            1,
            status="invalid",
            e=None,
            l_ms=None,
            v_mb=None,
            s_mb=None,
            gpu_seconds=4.0,
            failure_stage="build",
            failure_code="engine_build_failed",
        ),
        _row(2, status="running", e=None, l_ms=None, v_mb=None, s_mb=None, gpu_seconds=None, committed_at=None),
    ]
    blocks = [{"precision": "fp32", "channels": 32, "op": "standard"}, {"precision": "fp16", "channels": 64}]
    genomes = {"c0": {"n_blocks": 2, "stem_precision": "fp32", "head_precision": "fp32", "blocks": blocks}}
    columns, measures = export._measures(rows, genomes, frozenset({"c0"}), True, False)
    assert [m["attempt_id"] for m in measures] == ["c0.a1"]  # only what was measured is a measure
    assert not any(c.endswith("_SINTETICO") for c in columns)
    text = export._csv_bytes(columns, measures).decode("utf-8")
    [line] = list(csv.DictReader(text.splitlines()))
    assert line["E_erro_top1"] == "0.123456789012" and line["gpu_s"] == "61.0"  # full value
    assert line["precision_por_bloco"] == "fp32 fp16" and line["channels_por_bloco"] == "32 64"
    assert line["op_por_bloco"] == "standard NA" and line["stem_channels"] == ""  # not recorded: empty
    assert line["na_frente_mais_recente"] == "true" and line["preliminar"] == "true" and line["sintetico"] == "false"

    columns, attempts = export._attempts(rows, True, False)
    lines = list(csv.DictReader(export._csv_bytes(columns, attempts).decode("utf-8").splitlines()))
    assert [(a["status"], a["failure_stage"], a["failure_code"], a["gpu_s"]) for a in lines] == [
        ("valid", "", "", "61.0"),
        ("invalid", "build", "engine_build_failed", "4.0"),
        ("running", "", "", ""),
    ]
    columns, generations = export._generations(rows, True, False)
    [g] = generations
    assert (g["validas"], g["invalidas"], g["em_curso"], g["menor_E_erro_top1"]) == (1, 1, 1, 0.123456789012)

    mixed = export._value_columns(["gpu_s"], True, True)
    assert mixed == ["gpu_s", "gpu_s_SINTETICO"] and export._value_columns(["gpu_s"], False, False) == ["gpu_s"]


def test_cli_export_reads_any_schema_and_never_touches_the_database(
    run_db: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, run_id = run_db
    old = make_v1_copy(db, tmp_path / "v1" / "demo.sqlite")
    before = _digest(old)
    dest = tmp_path / "export-v1"
    base = ["export", "--db", str(old), "--run", run_id, "--dest", str(dest)]
    assert cli.main(base) == EXIT_REFUSED and not dest.exists()
    assert "error: ExportError" in capsys.readouterr().err

    assert cli.main([*base, "--allow-synthetic"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC results exported in labeled columns" in out and f"export {dest}: run {run_id}, 7 file(s)" in out
    assert "medidas.csv: 3 row(s) sha256=" in out
    assert _digest(old) == before and export.verify_export(dest) == []
    manifest = json.loads((dest / export.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1 and manifest["database_file"] == "demo.sqlite"

    assert cli.main([*base, "--allow-synthetic"]) == EXIT_REFUSED  # destination exists now
    assert "already exists" in capsys.readouterr().err
    missing = cli.main(
        ["export", "--db", str(tmp_path / "none.sqlite"), "--run", run_id, "--dest", str(tmp_path / "x")]
    )
    assert missing == EXIT_REFUSED and not (tmp_path / "none.sqlite").exists() and not (tmp_path / "x").exists()
    assert "export" in cli.COMMANDS and "cnnlab-tcc export" in (cli.__doc__ or "")
