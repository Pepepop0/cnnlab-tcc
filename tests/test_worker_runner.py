"""Testes de worker/runner.py.

`load_data`, `train` e `run_ga` sao mockados: o loop da fila inteiro roda em
milissegundos, sem GPU e sem dataset. O disco e real, redirecionado para tmp.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import torch

from cnnlab.config import (
    ClassicJob,
    DatasetConfig,
    GAConfig,
    GAJob,
    ModelConfig,
    TrainConfig,
)
from cnnlab.core.data import DataBundle
from cnnlab.core.ga.engine import GAResult
from cnnlab.core.trainer import TrainResult
from cnnlab.events import (
    EpochEnd,
    RunCancelled,
    RunFailed,
    RunFinished,
    RunStarted,
)
from cnnlab.storage import layout, reader
from cnnlab.worker import runner as runner_mod
from cnnlab.worker.queue import JobQueue
from cnnlab.worker.runner import Runner

CPU = torch.device("cpu")


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redireciona runs/ e queue.json para um diretorio temporario."""
    monkeypatch.setattr(layout, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(layout, "QUEUE_FILE", tmp_path / "queue.json")
    monkeypatch.setattr("cnnlab.worker.queue.QUEUE_FILE", tmp_path / "queue.json")
    yield tmp_path


@pytest.fixture
def bundle() -> DataBundle:
    return DataBundle(
        x_train=torch.zeros(4, 3, 32, 32),
        y_train=torch.zeros(4, dtype=torch.int64),
        x_val=torch.zeros(2, 3, 32, 32),
        y_val=torch.zeros(2, dtype=torch.int64),
        class_names=["apple", "orange"],
        num_classes=2,
    )


@pytest.fixture
def fake_load(monkeypatch: pytest.MonkeyPatch, bundle: DataBundle) -> list[DatasetConfig]:
    """Substitui load_data e registra cada config pedida."""
    calls: list[DatasetConfig] = []

    def load(cfg: DatasetConfig, device: torch.device) -> DataBundle:
        calls.append(cfg)
        return bundle

    monkeypatch.setattr(runner_mod, "load_data", load)
    return calls


class Collector:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)

    def of(self, kind) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def classic_job(epochs: int = 2, note: str = "") -> ClassicJob:
    return ClassicJob(
        dataset=DatasetConfig(class_subset=2),
        train=TrainConfig(model=ModelConfig(base_filters=8, n_blocks=2), epochs=epochs),
        note=note,
    )


def ga_job(**kwargs) -> GAJob:
    base = {"population_size": 4, "generations": 2, "elitism": 1, "final_epochs": 5}
    base.update(kwargs)
    return GAJob(dataset=DatasetConfig(class_subset=2), ga=GAConfig(**base))


def train_result(cancelled: bool = False, weights: bool = True) -> TrainResult:
    return TrainResult(
        best_val_acc=0.42,
        final_val_acc=0.40,
        final_train_loss=1.2,
        epochs_run=2,
        elapsed_s=1.5,
        cancelled=cancelled,
        state_dict={"0.weight": torch.zeros(2, 2)} if weights else None,
    )


def ga_result(cancelled: bool = False, weights: bool = True, final_val_acc: float | None = 0.77) -> GAResult:
    return GAResult(
        final_val_acc=None if cancelled else final_val_acc,
        best_genome={
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "dropout": 0.2,
            "batch_size": 64,
            "optimizer": "adam",
            "base_filters": 32,
            "n_blocks": 2,
        },
        best_fitness=0.31,
        best_state_dict={"0.weight": torch.zeros(2, 2)} if weights else None,
        generations_run=2,
        total_trainings=7,
        cancelled=cancelled,
    )


def stub_train(monkeypatch: pytest.MonkeyPatch, result: TrainResult) -> list[dict]:
    calls: list[dict] = []

    def fake(train_cfg, data, ctx, on_event, cancel, device, keep_weights=False):
        calls.append({"cfg": train_cfg, "ctx": ctx, "keep_weights": keep_weights})
        for epoch in range(1, result.epochs_run + 1):
            on_event(
                EpochEnd(
                    run_id=ctx.run_id,
                    epoch=epoch,
                    epochs_total=result.epochs_run,
                    train_loss=1.0,
                    val_loss=1.0,
                    val_acc=0.1 * epoch,
                    elapsed_s=0.01,
                    unit_index=ctx.unit_index,
                    unit_total=ctx.unit_total,
                    gen=ctx.gen,
                    individual=ctx.individual,
                )
            )
        return result

    monkeypatch.setattr(runner_mod, "train", fake)
    return calls


def stub_run_ga(monkeypatch: pytest.MonkeyPatch, result: GAResult) -> list[str]:
    calls: list[str] = []

    def fake(job, data, run_id, on_event, cancel, device):
        calls.append(run_id)
        on_event(RunStarted(run_id=run_id, kind="ga", unit_total=7, epochs_per_unit=3))
        if not result.cancelled:
            # Ultima epoca vista e a do retreino final: e ela que rotula o peso.
            on_event(
                EpochEnd(
                    run_id=run_id,
                    epoch=5,
                    epochs_total=5,
                    train_loss=0.5,
                    val_loss=0.9,
                    val_acc=0.77,
                    elapsed_s=0.01,
                    unit_index=6,
                    unit_total=7,
                    gen=1,
                    individual=0,
                )
            )
            on_event(
                RunFinished(
                    run_id=run_id,
                    kind="ga",
                    best_fitness=result.best_fitness,
                    best_genome=result.best_genome,
                    total_trainings=7,
                    total_epochs=21,
                    elapsed_s=9.0,
                    artifact="best.pt",
                )
            )
        return result

    monkeypatch.setattr(runner_mod, "run_ga", fake)
    return calls


def status_of(run_id: str) -> dict:
    path = layout.run_dir(run_id) / layout.STATUS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def only_run_id() -> str:
    return next(p.name for p in layout.RUNS_DIR.iterdir() if p.is_dir())


# --- caminho feliz -----------------------------------------------------------


def test_classic_job_writes_run_and_marks_done(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_train(monkeypatch, train_result())
    queue = JobQueue()
    job = queue.add(classic_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    run_id = only_run_id()
    assert status_of(run_id)["status"] == "done"
    assert queue.next_pending() is None
    assert [e.run_id for e in events.of(RunStarted)] == [run_id]
    assert len(events.of(RunFinished)) == 1
    assert reader.artifact_path(run_id) is not None


def test_classic_runstarted_and_runfinished_come_from_the_runner(
    monkeypatch: pytest.MonkeyPatch, fake_load: list
) -> None:
    stub_train(monkeypatch, train_result())
    queue = JobQueue()
    job = queue.add(classic_job(epochs=3))
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    started = events.of(RunStarted)[0]
    finished = events.of(RunFinished)[0]
    assert (started.kind, started.unit_total, started.epochs_per_unit) == ("classic", 1, 3)
    assert finished.total_trainings == 1
    assert finished.best_genome is None
    assert finished.artifact == "best.pt"
    assert finished.best_fitness == 0.42


def test_classic_asks_for_weights(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    calls = stub_train(monkeypatch, train_result())
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(classic_job()), threading.Event())

    assert calls[0]["keep_weights"] is True
    assert calls[0]["ctx"].unit_total == 1


def test_ga_job_does_not_duplicate_the_terminal_events(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_run_ga(monkeypatch, ga_result())
    queue = JobQueue()
    job = queue.add(ga_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    # run_ga ja emitiu os dois; o runner nao pode emitir de novo.
    assert len(events.of(RunStarted)) == 1
    assert len(events.of(RunFinished)) == 1
    assert status_of(only_run_id())["status"] == "done"


def test_ga_artifact_uses_final_val_acc_not_best_fitness(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    """0.77 e o retreino final; 0.31 e a busca curta e subestimaria o peso."""
    stub_run_ga(monkeypatch, ga_result(final_val_acc=0.77))
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(ga_job()), threading.Event())

    saved = torch.load(reader.artifact_path(only_run_id()), weights_only=False)
    assert saved["val_acc"] == pytest.approx(0.77)
    assert saved["class_names"] == ["apple", "orange"]
    assert saved["model_config"]["base_filters"] == 32
    assert saved["model_config"]["n_blocks"] == 2


def test_ga_artifact_falls_back_to_best_fitness_without_final_val_acc(
    monkeypatch: pytest.MonkeyPatch, fake_load: list
) -> None:
    stub_run_ga(monkeypatch, ga_result(final_val_acc=None))
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(ga_job()), threading.Event())

    saved = torch.load(reader.artifact_path(only_run_id()), weights_only=False)
    assert saved["val_acc"] == pytest.approx(0.31)


def test_artifact_never_pickles_the_model(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_train(monkeypatch, train_result())
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(classic_job()), threading.Event())

    saved = torch.load(reader.artifact_path(only_run_id()), weights_only=False)
    assert set(saved) == {"state_dict", "model_config", "class_names", "val_acc", "run_id"}


# --- falhas ------------------------------------------------------------------


def test_exception_marks_failed_and_emits_runfailed(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("cuda derreteu")

    monkeypatch.setattr(runner_mod, "train", boom)
    queue = JobQueue()
    job = queue.add(classic_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    failed = events.of(RunFailed)
    assert len(failed) == 1
    assert "cuda derreteu" in failed[0].error
    assert "RuntimeError" in failed[0].error
    assert "Traceback" in failed[0].traceback
    assert status_of(only_run_id())["status"] == "failed"


def test_one_bad_job_does_not_stop_the_queue(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    """Requisito central do modo Sleep."""
    attempts: list[int] = []

    def flaky(train_cfg, data, ctx, on_event, cancel, device, keep_weights=False):
        attempts.append(len(attempts))
        if len(attempts) == 2:
            raise ValueError("job do meio explodiu")
        return train_result()

    monkeypatch.setattr(runner_mod, "train", flaky)
    queue = JobQueue()
    queue.add_many([classic_job(note="a"), classic_job(note="b"), classic_job(note="c")])

    stop = threading.Event()
    events = Collector()
    runner = Runner(queue, events, CPU)

    original = queue.next_pending

    def next_pending_then_stop():
        job = original()
        if job is None:
            stop.set()
        return job

    monkeypatch.setattr(queue, "next_pending", next_pending_then_stop)
    runner.run_forever(stop, threading.Event())

    assert len(attempts) == 3
    statuses = sorted(status_of(p.name)["status"] for p in layout.RUNS_DIR.iterdir())
    assert statuses == ["done", "done", "failed"]
    assert queue.pending_count() == 0


def test_failure_before_the_writer_exists_does_not_kill_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_writer(*args, **kwargs):
        raise OSError("disco cheio")

    monkeypatch.setattr(runner_mod, "RunWriter", broken_writer)
    queue = JobQueue()
    job = queue.add(classic_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    assert len(events.of(RunFailed)) == 1
    assert "disco cheio" in events.of(RunFailed)[0].error


# --- cancelamento ------------------------------------------------------------


def test_cancelled_ga_emits_runcancelled_not_runfinished(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_run_ga(monkeypatch, ga_result(cancelled=True, weights=False))
    queue = JobQueue()
    job = queue.add(ga_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    cancelled = events.of(RunCancelled)
    assert len(cancelled) == 1
    assert cancelled[0].at_unit == 7  # GAResult.total_trainings
    assert events.of(RunFinished) == []
    assert status_of(only_run_id())["status"] == "cancelled"


def test_cancelled_classic_reports_its_unit_index(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_train(monkeypatch, train_result(cancelled=True))
    queue = JobQueue()
    job = queue.add(classic_job())
    events = Collector()

    Runner(queue, events, CPU).run_one(job, threading.Event())

    assert events.of(RunCancelled)[0].at_unit == 0
    assert events.of(RunFinished) == []


def test_cancelled_job_is_marked_cancelled_in_the_queue(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_run_ga(monkeypatch, ga_result(cancelled=True, weights=False))
    queue = JobQueue()
    job = queue.add(ga_job())

    Runner(queue, Collector(), CPU).run_one(job, threading.Event())

    payload = json.loads(layout.QUEUE_FILE.read_text(encoding="utf-8"))
    entry = next(j for j in payload["jobs"] if j["id"] == job.id)
    assert entry["status"] == "cancelled"
    assert entry["run_id"] == only_run_id()


def test_cancelled_run_has_no_summary_but_still_lists(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    """Escolha documentada: sem `RunFinished` nao ha `summary.json`."""
    stub_run_ga(monkeypatch, ga_result(cancelled=True, weights=False))
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(ga_job()), threading.Event())

    run_id = only_run_id()
    assert not (layout.run_dir(run_id) / layout.SUMMARY_FILE).exists()
    assert reader.artifact_path(run_id) is None

    listed = reader.list_runs()
    assert [r.run_id for r in listed] == [run_id]
    assert listed[0].status == "cancelled"
    assert listed[0].best_fitness is None


def test_cancelled_classic_keeps_the_weights_it_already_earned(
    monkeypatch: pytest.MonkeyPatch, fake_load: list
) -> None:
    stub_train(monkeypatch, train_result(cancelled=True))
    queue = JobQueue()
    Runner(queue, Collector(), CPU).run_one(queue.add(classic_job()), threading.Event())

    assert reader.artifact_path(only_run_id()) is not None


def test_cancel_current_does_not_stop_the_queue(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    seen_cancelled: list[bool] = []

    def fake(train_cfg, data, ctx, on_event, cancel, device, keep_weights=False):
        seen_cancelled.append(cancel.is_set())
        return train_result(cancelled=cancel.is_set())

    monkeypatch.setattr(runner_mod, "train", fake)
    queue = JobQueue()
    queue.add_many([classic_job(note="a"), classic_job(note="b")])

    stop = threading.Event()
    cancel_current = threading.Event()
    cancel_current.set()  # aborta o primeiro job

    original = queue.next_pending

    def next_pending_then_stop():
        job = original()
        if job is None:
            stop.set()
        return job

    monkeypatch.setattr(queue, "next_pending", next_pending_then_stop)
    Runner(queue, Collector(), CPU).run_forever(stop, cancel_current)

    # O primeiro viu o cancelamento; o segundo comecou limpo.
    assert seen_cancelled == [True, False]
    statuses = sorted(status_of(p.name)["status"] for p in layout.RUNS_DIR.iterdir())
    assert statuses == ["cancelled", "done"]


# --- loop --------------------------------------------------------------------


def test_three_jobs_run_in_sequence(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    stub_train(monkeypatch, train_result())
    queue = JobQueue()
    queue.add_many([classic_job(note=n) for n in "abc"])

    stop = threading.Event()
    original = queue.next_pending

    def next_pending_then_stop():
        job = original()
        if job is None:
            stop.set()
        return job

    monkeypatch.setattr(queue, "next_pending", next_pending_then_stop)
    Runner(queue, Collector(), CPU).run_forever(stop, threading.Event())

    assert queue.pending_count() == 0
    assert len(list(layout.RUNS_DIR.iterdir())) == 3


def test_stop_ends_the_loop_after_the_current_job(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    def fake(train_cfg, data, ctx, on_event, cancel, device, keep_weights=False):
        stop.set()  # para o loop enquanto este job roda
        return train_result()

    monkeypatch.setattr(runner_mod, "train", fake)
    queue = JobQueue()
    queue.add_many([classic_job(note="a"), classic_job(note="b")])
    stop = threading.Event()

    Runner(queue, Collector(), CPU).run_forever(stop, threading.Event())

    # O job em andamento terminou; o segundo nem comecou.
    assert queue.pending_count() == 1
    assert len(list(layout.RUNS_DIR.iterdir())) == 1


def test_empty_queue_waits_instead_of_busy_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    stop = threading.Event()

    class SpyStop:
        def is_set(self) -> bool:
            return len(waits) >= 3

        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            return False

    Runner(JobQueue(), Collector(), CPU).run_forever(SpyStop(), threading.Event())

    assert waits == [2.0, 2.0, 2.0]  # espera bloqueante, nao giro de CPU


def test_dataset_is_loaded_once_for_jobs_sharing_a_config(monkeypatch: pytest.MonkeyPatch, bundle: DataBundle) -> None:
    """O cache de `core.data.load_data` e o que segura isso."""
    from cnnlab.core import data as data_mod

    constructed: list[int] = []

    class FakeCIFAR100:
        def __init__(self, root: str, train: bool = True, download: bool = False) -> None:
            constructed.append(1)
            import numpy as np

            self.data = np.zeros((40, 32, 32, 3), dtype=np.uint8)
            self.targets = [i % 2 for i in range(40)]
            self.classes = ["apple", "orange"]

    data_mod.clear_data_cache()
    monkeypatch.setattr(data_mod, "CIFAR100", FakeCIFAR100)
    stub_train(monkeypatch, train_result())

    queue = JobQueue()
    queue.add_many([classic_job(note="a"), classic_job(note="b"), classic_job(note="c")])

    stop = threading.Event()
    original = queue.next_pending

    def next_pending_then_stop():
        job = original()
        if job is None:
            stop.set()
        return job

    monkeypatch.setattr(queue, "next_pending", next_pending_then_stop)
    Runner(queue, Collector(), CPU).run_forever(stop, threading.Event())
    data_mod.clear_data_cache()

    assert len(constructed) == 1  # tres jobs, um unico carregamento


def test_memory_is_released_between_jobs(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    collected: list[str] = []
    monkeypatch.setattr(runner_mod.gc, "collect", lambda: collected.append("gc"))
    stub_train(monkeypatch, train_result())

    queue = JobQueue()
    queue.add_many([classic_job(note="a"), classic_job(note="b")])
    stop = threading.Event()
    original = queue.next_pending

    def next_pending_then_stop():
        job = original()
        if job is None:
            stop.set()
        return job

    monkeypatch.setattr(queue, "next_pending", next_pending_then_stop)
    Runner(queue, Collector(), CPU).run_forever(stop, threading.Event())

    assert len(collected) == 2  # um por job


def test_crash_with_a_running_job_puts_it_back_to_pending(monkeypatch: pytest.MonkeyPatch, fake_load: list) -> None:
    """Aceite 3: matar o processo com job `running` e reabrir."""
    queue = JobQueue()
    job = queue.add(classic_job())
    queue.mark(job.id, "running")

    revived = JobQueue()
    revived.load()

    assert revived.next_pending().id == job.id
