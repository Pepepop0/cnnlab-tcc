"""Testes de storage/. Nao exigem GPU nem torch (o teste de artefato pula)."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from cnnlab.config import ClassicJob, GAConfig, GAJob
from cnnlab.events import (
    EpochEnd,
    GenerationDone,
    IndividualDone,
    RunCancelled,
    RunFailed,
    RunFinished,
    RunStarted,
)
from cnnlab.storage import layout, reader, writer
from cnnlab.storage.writer import RunWriter


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redireciona RUNS_DIR para tmp e zera o registro de runs vivas."""
    root = tmp_path / "runs"
    monkeypatch.setattr(layout, "RUNS_DIR", root)
    monkeypatch.setattr(layout, "_ACTIVE_RUNS", set())
    return root


def _ga_spec(note: str = "nota ga") -> GAJob:
    return GAJob(ga=GAConfig(population_size=4, generations=2), note=note)


def _write_status(root: Path, run_id: str, **fields: object) -> None:
    """Escreve um status.json a mao, sem passar pelo writer."""
    payload: dict[str, object] = {
        "run_id": run_id,
        "kind": run_id.split("_")[1],
        "status": "done",
        "created_at": "2026-09-01T23:14:02",
        "started_at": "2026-09-01T23:14:03",
        "finished_at": None,
        "error": None,
        "note": "",
    }
    payload.update(fields)
    layout.atomic_write_json(root / run_id / layout.STATUS_FILE, payload)


def _status_on_disk(root: Path, run_id: str) -> str:
    data = json.loads((root / run_id / layout.STATUS_FILE).read_text(encoding="utf-8"))
    return data["status"]


# -- layout --------------------------------------------------------------


def test_new_run_id_formato() -> None:
    run_id = layout.new_run_id("ga", now=datetime(2026, 9, 1, 23, 14))
    assert re.fullmatch(r"20260901-2314_ga_[0-9a-f]{4}", run_id)


def test_new_run_id_nao_colide_no_mesmo_minuto() -> None:
    now = datetime(2026, 9, 1, 23, 14)
    ids = {layout.new_run_id("classic", now=now) for _ in range(50)}
    assert len(ids) > 45  # 4 hex = 65536 valores; colisao aqui seria bug


def test_run_dir_segue_runs_dir_redirecionado(runs_root: Path) -> None:
    assert layout.run_dir("abc") == runs_root / "abc"


def test_atomic_write_json_nao_deixa_tmp(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "status.json"
    layout.atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert list(path.parent.glob("*.tmp")) == []


def test_atomic_write_json_sobrescreve(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    layout.atomic_write_json(path, {"status": "running"})
    layout.atomic_write_json(path, {"status": "done"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "done"}


def test_atomic_write_json_concorrente_nao_colide(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    workers = 16
    barrier = threading.Barrier(workers)
    payloads = [{"writer": index, "values": [index] * 2_000} for index in range(workers)]

    def write(payload: dict[str, object]) -> None:
        barrier.wait()
        layout.atomic_write_json(path, payload)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(write, payload) for payload in payloads]
        for future in futures:
            future.result()

    final = json.loads(path.read_text(encoding="utf-8"))
    assert final in payloads
    assert list(path.parent.glob("*.tmp")) == []


def test_atomic_write_json_remove_temporario_em_erro(tmp_path: Path) -> None:
    path = tmp_path / "status.json"

    with pytest.raises(TypeError):
        layout.atomic_write_json(path, {"not_json": object()})

    assert not path.exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_append_jsonl_acrescenta(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    layout.append_jsonl(path, {"epoch": 1})
    layout.append_jsonl(path, {"epoch": 2})
    linhas = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["epoch"] for x in linhas] == [1, 2]


# -- writer --------------------------------------------------------------


def test_init_cria_pasta_e_config_antes_de_treinar(runs_root: Path) -> None:
    RunWriter("20260901-2314_ga_a3f1", _ga_spec())

    run = runs_root / "20260901-2314_ga_a3f1"
    assert (run / layout.CONFIG_FILE).is_file()
    assert (run / layout.STATUS_FILE).is_file()

    status = json.loads((run / layout.STATUS_FILE).read_text(encoding="utf-8"))
    assert status["status"] == "queued"
    assert status["kind"] == "ga"
    assert status["note"] == "nota ga"


def test_config_json_faz_roundtrip_de_jobspec(runs_root: Path) -> None:
    casos = (
        ("20260901-2314_ga_a3f1", _ga_spec()),
        ("20260901-2315_classic_b7c2", ClassicJob(note="classica")),
    )
    for run_id, spec in casos:
        RunWriter(run_id, spec)
        assert reader.read_config(run_id) == spec


def test_fluxo_completo_grava_todos_os_arquivos(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    w = RunWriter(run_id, _ga_spec())
    w.handle(RunStarted(run_id=run_id, kind="ga", unit_total=8, epochs_per_unit=3))

    for epoch in (1, 2):
        w.handle(
            EpochEnd(
                run_id=run_id,
                epoch=epoch,
                epochs_total=3,
                train_loss=3.9,
                val_loss=3.7,
                val_acc=0.12,
                elapsed_s=4.2,
                unit_index=0,
                unit_total=8,
                gen=0,
                individual=0,
            )
        )
    w.handle(
        IndividualDone(
            run_id=run_id,
            gen=0,
            individual=0,
            genome={"lr": 0.003},
            fitness=0.14,
            epochs=3,
            elapsed_s=12.6,
        )
    )
    w.handle(
        GenerationDone(
            run_id=run_id,
            gen=0,
            best_fitness=0.14,
            mean_fitness=0.11,
            std_fitness=0.01,
            best_genome={"lr": 0.003},
            elapsed_s=94.3,
        )
    )
    w.handle(
        RunFinished(
            run_id=run_id,
            kind="ga",
            best_fitness=0.17,
            best_genome={"lr": 0.003},
            total_trainings=8,
            total_epochs=24,
            elapsed_s=1841.2,
            artifact="best.pt",
        )
    )

    assert len(reader.read_progress(run_id)) == 2
    assert len(reader.read_individuals(run_id)) == 1
    assert len(reader.read_generations(run_id)) == 1
    assert reader.read_progress(run_id)[0]["event"] == "EpochEnd"
    assert reader.read_progress(run_id)[0]["unit_total"] == 8

    run = runs_root / run_id
    status = json.loads((run / layout.STATUS_FILE).read_text(encoding="utf-8"))
    summary = json.loads((run / layout.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert status["status"] == "done"
    assert status["started_at"] is not None
    assert status["finished_at"] is not None
    assert summary["best_fitness"] == 0.17
    assert summary["total_epochs"] == 24
    assert summary["artifact"] == "best.pt"


def test_start_marca_running(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    w = RunWriter(run_id, _ga_spec())
    w.start()
    status = json.loads((runs_root / run_id / layout.STATUS_FILE).read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["started_at"] is not None


def test_fail_grava_status_failed_com_erro(runs_root: Path) -> None:
    run_id = "20260901-2314_classic_b7c2"
    w = RunWriter(run_id, ClassicJob())
    w.start()
    w.fail(RuntimeError("CUDA out of memory"))

    status = json.loads((runs_root / run_id / layout.STATUS_FILE).read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "CUDA out of memory" in status["error"]

    # fail() seguido de handle(RunFailed) e o fluxo do runner: nao pode quebrar
    w.handle(RunFailed(run_id=run_id, error="CUDA out of memory", traceback="..."))
    assert _status_on_disk(runs_root, run_id) == "failed"


def test_cancelamento_grava_status_cancelled(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    w = RunWriter(run_id, _ga_spec())
    w.start()
    w.handle(RunCancelled(run_id=run_id, at_unit=3))
    assert _status_on_disk(runs_root, run_id) == "cancelled"


def test_writer_nao_importa_torch_no_topo() -> None:
    assert not hasattr(writer, "torch")


def test_save_artifact_grava_best_pt(runs_root: Path) -> None:
    torch = pytest.importorskip("torch")
    run_id = "20260901-2314_classic_b7c2"
    spec = ClassicJob()
    w = RunWriter(run_id, spec)

    path = w.save_artifact({"w": torch.zeros(2)}, spec.train.model, ["apple", "orange"], 0.1731)
    assert path == runs_root / run_id / layout.ARTIFACT_FILE
    assert reader.artifact_path(run_id) == path
    assert list((runs_root / run_id).glob("*.tmp")) == []

    blob = torch.load(path, weights_only=False)
    assert blob["run_id"] == run_id
    assert blob["class_names"] == ["apple", "orange"]
    assert blob["model_config"] == spec.train.model.model_dump()


def test_artifact_path_none_sem_arquivo(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    RunWriter(run_id, _ga_spec())
    assert reader.artifact_path(run_id) is None


# -- reader --------------------------------------------------------------


def test_read_jsonl_tolera_ultima_linha_truncada(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    path = runs_root / run_id / layout.PROGRESS_FILE
    layout.append_jsonl(path, {"epoch": 1})
    layout.append_jsonl(path, {"epoch": 2})
    with path.open("a", encoding="utf-8") as fh:  # crash no meio de um write
        fh.write('{"epoch": 3, "val_ac')

    assert [x["epoch"] for x in reader.read_progress(run_id)] == [1, 2]


def test_read_jsonl_ausente_retorna_vazio(runs_root: Path) -> None:
    assert reader.read_progress("nao-existe") == []
    assert reader.read_generations("nao-existe") == []
    assert reader.read_individuals("nao-existe") == []


def test_list_runs_vazio_sem_diretorio(runs_root: Path) -> None:
    assert reader.list_runs() == []


def test_list_runs_ordena_por_created_at_desc(runs_root: Path) -> None:
    _write_status(runs_root, "20260901-2314_ga_a3f1", created_at="2026-09-01T23:14:02")
    _write_status(runs_root, "20260901-2350_classic_b7c2", created_at="2026-09-01T23:50:00")
    _write_status(runs_root, "20260830-1000_ga_c9d3", created_at="2026-08-30T10:00:00")

    ids = [r.run_id for r in reader.list_runs()]
    assert ids == [
        "20260901-2350_classic_b7c2",
        "20260901-2314_ga_a3f1",
        "20260830-1000_ga_c9d3",
    ]


def test_list_runs_filtra_por_kind(runs_root: Path) -> None:
    _write_status(runs_root, "20260901-2314_ga_a3f1")
    _write_status(runs_root, "20260901-2350_classic_b7c2")

    assert [r.run_id for r in reader.list_runs(kind="ga")] == ["20260901-2314_ga_a3f1"]
    assert [r.run_id for r in reader.list_runs(kind="classic")] == ["20260901-2350_classic_b7c2"]
    assert len(reader.list_runs()) == 2


def test_list_runs_le_summary_para_fitness_e_elapsed(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    _write_status(runs_root, run_id, status="done", note="experimento 1")
    layout.atomic_write_json(
        runs_root / run_id / layout.SUMMARY_FILE,
        {"best_fitness": 0.1731, "elapsed_s": 1841.2},
    )

    (item,) = reader.list_runs()
    assert item.status == "done"
    assert item.best_fitness == pytest.approx(0.1731)
    assert item.elapsed_s == pytest.approx(1841.2)
    assert item.note == "experimento 1"


def test_list_runs_sem_summary_tem_campos_none(runs_root: Path) -> None:
    _write_status(runs_root, "20260901-2314_ga_a3f1", status="queued")
    (item,) = reader.list_runs()
    assert item.status == "queued"
    assert item.best_fitness is None
    assert item.elapsed_s is None


def test_list_runs_marca_run_orfa_como_failed_sem_reescrever(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    _write_status(runs_root, run_id, status="running")

    (item,) = reader.list_runs()
    assert item.status == "failed"
    # o disco continua dizendo running: o reader nunca corrige estado
    assert _status_on_disk(runs_root, run_id) == "running"


def test_list_runs_mantem_running_de_writer_vivo(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    w = RunWriter(run_id, _ga_spec())
    w.start()
    (item,) = reader.list_runs()
    assert item.status == "running"


def test_list_runs_ignora_pasta_sem_metadados(runs_root: Path) -> None:
    (runs_root / "lixo").mkdir(parents=True)
    layout.append_jsonl(runs_root / "20260901-2314_ga_a3f1" / layout.PROGRESS_FILE, {"epoch": 1})
    assert reader.list_runs() == []


def test_list_runs_tolera_status_corrompido(runs_root: Path) -> None:
    run_id = "20260901-2314_ga_a3f1"
    path = runs_root / run_id / layout.STATUS_FILE
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id": "2026', encoding="utf-8")
    layout.atomic_write_json(runs_root / run_id / layout.SUMMARY_FILE, {"best_fitness": 0.5})

    (item,) = reader.list_runs()
    assert item.kind == "ga"  # deduzido do run_id
    assert item.status == "done"
    assert item.created_at == "2026-09-01T23:14:00"
    assert item.best_fitness == pytest.approx(0.5)


def test_list_runs_nao_le_jsonl(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "20260901-2314_ga_a3f1"
    _write_status(runs_root, run_id)
    layout.append_jsonl(runs_root / run_id / layout.PROGRESS_FILE, {"epoch": 1})

    def explode(path: Path) -> list[dict[str, object]]:
        raise AssertionError("list_runs nao pode ler " + str(path))

    monkeypatch.setattr(reader, "_read_jsonl", explode)
    assert len(reader.list_runs()) == 1


def test_list_runs_50_runs_abaixo_de_100ms(runs_root: Path) -> None:
    for i in range(50):
        run_id = "202609%02d-2314_ga_%04x" % (i % 28 + 1, i)
        _write_status(runs_root, run_id, status="done")
        layout.atomic_write_json(
            runs_root / run_id / layout.SUMMARY_FILE,
            {"best_fitness": 0.5, "elapsed_s": 10.0},
        )

    inicio = time.perf_counter()
    runs = reader.list_runs()
    decorrido = time.perf_counter() - inicio
    assert len(runs) == 50
    assert decorrido < 0.1


def test_read_config_ausente_levanta(runs_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        reader.read_config("20260901-2314_ga_a3f1")
