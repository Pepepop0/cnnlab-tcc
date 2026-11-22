"""Queue file of the demo: editing rules, sleep draws, lock, spec and stop flag."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from cnnlab.tcc.operation import schedule as s
from cnnlab.tcc.operation.demo import DemoRunSpec

NOW = datetime(2026, 9, 29, 22, 30)


def _queue(*names: str) -> s.QueueState:
    state = s.QueueState()
    for name in names:
        state = s.add_item(state, name, s.RunSettings(), now=NOW)
    return state


def test_settings_validate_limits_and_types() -> None:
    s.RunSettings(population=64, generations=50, train_per_class=450)
    with pytest.raises(s.ScheduleError, match="populacao|population"):
        s.RunSettings(population=1)
    with pytest.raises(s.ScheduleError, match="inteiro"):
        s.RunSettings(generations=2.5)  # type: ignore[arg-type]
    with pytest.raises(s.ScheduleError, match="recorte"):
        s.RunSettings(slice="int8")


def test_method_and_the_two_optional_settings_are_validated_and_survive_the_file(tmp_path: Path) -> None:
    assert s.RunSettings().method == "joint" and s.METHODS == ("joint", "mono", "random", "sequential")
    with pytest.raises(s.ScheduleError, match="metodo"):
        s.RunSettings(method="nsga3")
    # the sequential method is limited by GPU time per stage and needs a slice it can compress in
    with pytest.raises(s.ScheduleError, match="limite de geracoes"):
        s.RunSettings(method="sequential", slice="fp16_block")  # generations default to 2
    with pytest.raises(s.ScheduleError, match="fp16_regions"):
        s.RunSettings(method="sequential", generations=None)  # slice fp32 pins every representation gene
    for richer in ("fp16_block", "fp16_regions", "admitted"):  # admitted: the campaign's slice (G7)
        assert s.RunSettings(method="sequential", slice=richer, generations=None).slice == richer
    assert s.SLEEP_SLICES == ("fp32", "fp16_block", "fp16_regions")  # the sleep mode keeps to the original three
    with pytest.raises(s.ScheduleError, match="sono"):
        s.SleepConfig(slices=("admitted",))
    campaign = s.RunSettings(
        method="random", population=30, generations=None, train_per_class=None, budget_gpu_s=19_800
    )
    assert campaign.generations is None and campaign.train_per_class is None
    for name in ("population", "epochs", "budget_gpu_s", "search_seed"):
        with pytest.raises(s.ScheduleError, match=name):
            s.RunSettings(**{name: None})  # None is for the two optional settings only
    state = s.mutate(tmp_path, lambda st: s.add_item(st, "campanha", campaign, now=NOW))
    assert s.load_state(tmp_path) == state and s.load_state(tmp_path).items[0].settings == campaign
    spec = s.spec_payload(state.items[0])
    assert spec["method"] == "random"
    assert spec["factory_args"]["generations"] is None and spec["factory_args"]["train_per_class"] is None
    assert spec["factory_args"]["population"] == 30


def test_a_queue_file_written_before_the_method_existed_still_loads(tmp_path: Path) -> None:
    state = s.add_item(s.QueueState(), "antiga", s.RunSettings(), now=NOW)
    data = s.state_to_json(state)
    del data["items"][0]["settings"]["method"]
    (tmp_path / s.QUEUE_FILE).write_text(json.dumps(data), encoding="utf-8")
    loaded = s.load_state(tmp_path)
    assert loaded.items[0].settings.method == "joint" and loaded.items[0].settings == s.RunSettings()


def test_add_assigns_ids_and_refuses_bad_or_duplicated_names() -> None:
    state = _queue("a", "b")
    assert [(i.item_id, i.name, i.state, i.tag) for i in state.items] == [
        ("0001", "a", s.PLANNED, s.MANUAL),
        ("0002", "b", s.PLANNED, s.MANUAL),
    ]
    with pytest.raises(s.ScheduleError, match="ja existe"):
        s.add_item(state, "A", s.RunSettings())
    with pytest.raises(s.ScheduleError, match="nome"):
        s.add_item(state, "com espaco", s.RunSettings())


def test_started_runs_cannot_be_edited_moved_or_deleted() -> None:
    state = s.set_item_state(_queue("a", "b"), "0001", s.RUNNING, run_id="r", job_id="j", now=NOW)
    for change in (
        lambda: s.update_item(state, "0001", "a", s.RunSettings()),
        lambda: s.delete_item(state, "0001"),
        lambda: s.move_item(state, "0001", 1),
    ):
        with pytest.raises(s.ScheduleError, match="ja saiu da fila"):
            change()
    edited = s.update_item(state, "0002", "b2", s.RunSettings(generations=9))
    assert edited.items[1].name == "b2" and edited.items[1].settings.generations == 9


def test_delete_planned_keeps_history_and_move_only_reorders_planned() -> None:
    state = s.set_item_state(_queue("a", "b", "c", "d"), "0002", s.DONE, now=NOW)
    moved = s.move_item(state, "0004", -5)
    assert [i.name for i in moved.items] == ["d", "b", "a", "c"]  # b (done) keeps its slot
    assert [i.name for i in s.delete_planned(moved).items] == ["b"]
    assert [i.name for i in s.delete_item(state, "0003").items] == ["a", "b", "d"]


def test_manual_runs_go_before_drawn_runs() -> None:
    state = s.replace(_queue(), sleep=s.SleepConfig(enabled=True, seed=3))
    state = s.add_sleep_draw(state, now=NOW)
    state = s.add_item(state, "manual", s.RunSettings(), now=NOW)
    assert s.next_planned(state).name == "manual"  # type: ignore[union-attr]
    state = s.set_item_state(state, state.items[1].item_id, s.RUNNING, now=NOW)
    assert s.next_planned(state).tag == s.SLEEP  # type: ignore[union-attr]


def test_sleep_draws_are_reproducible_and_inside_the_ranges() -> None:
    sleep = s.SleepConfig(
        enabled=True,
        population=(4, 6),
        generations=(2, 3),
        slices=("fp32", "fp16_block"),
        epochs=(3, 4),
        train_per_class=(50, 60),
        seed=11,
    )
    draws = [s.draw_settings(sleep, i) for i in range(40)]
    assert draws == [s.draw_settings(sleep, i) for i in range(40)]
    assert {d.slice for d in draws} == {"fp32", "fp16_block"}
    assert all(4 <= d.population <= 6 and 2 <= d.generations <= 3 and 50 <= d.train_per_class <= 60 for d in draws)
    assert s.draw_settings(sleep, 0) != s.draw_settings(s.replace(sleep, seed=12), 0)
    state = s.add_sleep_draw(s.add_sleep_draw(s.QueueState(sleep=sleep), now=NOW), now=NOW)
    assert state.sleep.draws == 2 and [i.name for i in state.items] == [
        "sono-20260929-s11-001",
        "sono-20260929-s11-002",
    ]


def test_sleep_ranges_are_validated() -> None:
    with pytest.raises(s.ScheduleError, match="faixa"):
        s.SleepConfig(population=(8, 4))
    with pytest.raises(s.ScheduleError, match="recortes"):
        s.SleepConfig(slices=())


def test_deadline_is_the_next_occurrence_and_ends_sleep() -> None:
    assert s.next_occurrence("08:00", NOW) == "2026-09-30T08:00:00"
    assert s.next_occurrence("23:00", NOW) == "2026-09-29T23:00:00"
    with pytest.raises(s.ScheduleError):
        s.next_occurrence("25:00", NOW)
    sleep = s.SleepConfig(enabled=True, until="2026-09-30T08:00:00")
    assert s.sleep_active(sleep, NOW) and not s.sleep_active(sleep, datetime(2026, 9, 30, 8, 1))
    assert not s.sleep_active(s.replace(sleep, enabled=False), NOW)
    assert s.sleep_active(s.SleepConfig(enabled=True), NOW)


def test_file_round_trip_and_atomic_mutate(tmp_path: Path) -> None:
    assert s.load_state(tmp_path) == s.QueueState()
    state = s.mutate(tmp_path, lambda st: s.add_item(st, "a", s.RunSettings(slice="fp16_block"), now=NOW))
    state = s.mutate(tmp_path, lambda st: s.replace(st, sleep=s.SleepConfig(enabled=True, slices=("fp32",))))
    assert s.load_state(tmp_path) == state
    assert not (tmp_path / s.LOCK_FILE).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_waits_breaks_stale_and_times_out(tmp_path: Path) -> None:
    lock = tmp_path / s.LOCK_FILE
    lock.write_text("123")
    with pytest.raises(s.ScheduleError, match="fila ocupada"):
        with s.locked(tmp_path, timeout_s=0.1):
            pass
    old = time.time() - 60
    os.utime(lock, (old, old))
    with s.locked(tmp_path, timeout_s=0.1):
        assert lock.exists()
    assert not lock.exists()


def test_spec_is_accepted_by_the_demo_loader(tmp_path: Path) -> None:
    state = s.add_item(
        s.QueueState(),
        "fp16-teste",
        s.RunSettings(slice="fp16_block", population=6, generations=3, budget_gpu_s=900),
        now=NOW,
    )
    path = s.write_spec(tmp_path, state.items[0])
    assert path.read_bytes()[:1] == b"{"  # no BOM
    spec = DemoRunSpec.load(path)
    assert spec.run_id == "fila-0001-fp16-teste" and spec.method == "joint" and spec.author == "Pedro Antunes"
    assert spec.budget_gpu_s == 900.0 and spec.factory == s.FACTORY
    assert dict(spec.factory_args) == {
        "train_per_class": 100,
        "epochs": 5,
        "population": 6,
        "generations": 3,
        "slice": "fp16_block",
    }


def test_stop_flag_turns_sleep_off_and_executor_heartbeat(tmp_path: Path) -> None:
    s.mutate(tmp_path, lambda st: s.replace(st, sleep=s.SleepConfig(enabled=True)))
    s.request_stop(tmp_path)
    assert s.stop_requested(tmp_path) and not s.load_state(tmp_path).sleep.enabled
    s.clear_stop(tmp_path)
    assert not s.stop_requested(tmp_path)
    assert s.read_executor(tmp_path) is None
    s.write_executor(tmp_path, pid=42, item_id="0001", message="rodando")
    assert s.read_executor(tmp_path)["alive"] is True  # type: ignore[index]
    data = json.loads((tmp_path / s.EXECUTOR_FILE).read_text(encoding="utf-8"))
    data["beat"] -= s.EXECUTOR_STALE_S + 1
    (tmp_path / s.EXECUTOR_FILE).write_text(json.dumps(data), encoding="utf-8")
    assert s.read_executor(tmp_path)["alive"] is False  # type: ignore[index]


def test_concurrent_writers_of_one_file_do_not_collide(tmp_path: Path) -> None:
    """Executor loop and heartbeat thread write executor.json at the same time (Windows sharing rules)."""
    import threading

    errors: list[BaseException] = []

    def writer(tag: str) -> None:
        try:
            for n in range(150):
                s.write_executor(tmp_path, pid=1, item_id=tag, message=str(n))
                s.read_executor(tmp_path)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and not list(tmp_path.glob("*.tmp"))


def test_population_must_be_even_on_add_and_edit_and_sleep_draws_only_even_sizes() -> None:
    with pytest.raises(s.ScheduleError, match="par"):
        s.add_item(s.QueueState(), "impar", s.RunSettings(population=5))
    state = _queue("a")
    with pytest.raises(s.ScheduleError, match="par"):
        s.update_item(state, "0001", "a", s.RunSettings(population=7))
    sizes = {s.draw_settings(s.SleepConfig(population=(3, 9), seed=4), i).population for i in range(60)}
    assert sizes == {4, 6, 8}
    assert s.draw_settings(s.SleepConfig(population=(5, 5)), 0).population == 6


def test_history_with_an_odd_population_still_loads(tmp_path: Path) -> None:
    """A run queued before the even rule (it failed in the worker) must not make the file unreadable."""
    state = s.add_item(s.QueueState(), "antiga", s.RunSettings(), now=NOW)
    old = s.replace(state.items[0], settings=s.RunSettings(population=5), state=s.FAILED)
    s.save_state(tmp_path, s.replace(state, items=(old,)))
    loaded = s.load_state(tmp_path)
    assert loaded.items[0].settings.population == 5 and loaded.items[0].state == s.FAILED
    s.mutate(tmp_path, lambda st: s.add_item(st, "nova", s.RunSettings(population=6), now=NOW))
    assert [i.name for i in s.load_state(tmp_path).items] == ["antiga", "nova"]
