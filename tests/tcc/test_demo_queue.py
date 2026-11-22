"""Fila tab of the demo TUI, driven headless (CPU only); the executor is always mocked."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, Checkbox, DataTable, Input, Static, TabbedContent

import cnnlab.tcc.demo_tui as tui
from cnnlab.tcc.demo_queue import QueuePane
from cnnlab.tcc.operation import schedule

pytestmark = pytest.mark.asyncio
SIZE = (170, 50)


@pytest.fixture
def qdir(tmp_path: Path) -> Path:
    return tmp_path / "fila"


@pytest.fixture
def fake_scheduler(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, Any], bool]]:
    """Stand-in for ``operation.scheduler`` (written by another agent); records every spawn."""
    import sys

    calls: list[tuple[Path, dict[str, Any], bool]] = []

    def spawn_executor(queue_dir: Path, **kw: Any) -> Any:
        calls.append((queue_dir, kw, schedule.stop_requested(queue_dir)))
        return types.SimpleNamespace(pid=4242, alive=False, log_path=queue_dir / "x.log")

    module = types.ModuleType("cnnlab.tcc.operation.scheduler")
    module.spawn_executor = spawn_executor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cnnlab.tcc.operation.scheduler", module)
    return calls


def _app(tmp_path: Path, qdir: Path, *, read_only: bool = False) -> tui.DemoTui:
    return tui.DemoTui(
        tmp_path / "db" / "tcc.sqlite3",
        tmp_path / "artifacts",
        queue_dir=qdir,
        read_only=read_only,
        refresh_s=0.2,
        extra_env={"CUDA_VISIBLE_DEVICES": ""},
    )


async def _press(pilot: Any, app: tui.DemoTui, button_id: str) -> None:
    app.query_one(f"#{button_id}", Button).press()
    await pilot.pause(0.1)


def _seed(qdir: Path, *names: str) -> None:
    for name in names:
        schedule.mutate(qdir, lambda s, n=name: schedule.add_item(s, n, schedule.RunSettings()))


def _names(qdir: Path) -> list[str]:
    return [i.name for i in schedule.load_state(qdir).items]


async def _set(pilot: Any, app: tui.DemoTui, widget_id: str, value: str) -> None:
    app.query_one(f"#{widget_id}", Input).value = value
    await pilot.pause()


async def test_add_item_from_form(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _set(pilot, app, "q-name", "minha-run")
        await _set(pilot, app, "q-population", "6")
        await _press(pilot, app, "q-add")
        item = schedule.load_state(qdir).items[0]
        assert (item.name, item.settings.population, item.state, item.tag) == ("minha-run", 6, "planejada", "manual")
        assert app.query_one("#q-table", DataTable).row_count == 1
        # the name suggestion follows the form: fp32-p6-g2-HHMM
        assert app.query_one("#q-name", Input).value.startswith("fp32-p6-g2-")


async def test_choose_the_method_and_the_two_campaign_settings_from_the_form(tmp_path: Path, qdir: Path) -> None:
    from textual.widgets import Select

    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        assert [value for _label, value in app.query_one("#q-method", Select)._options if value != Select.BLANK] == [
            "joint",
            "mono",
            "random",
            "sequential",
        ]
        assert app.query_one("#q-method", Select).value == "joint"
        app.query_one("#q-method", Select).value = "mono"
        await _set(pilot, app, "q-name", "campanha-mono")
        await _set(pilot, app, "q-population", "30")
        await _set(pilot, app, "q-generations", "sem limite")
        await _set(pilot, app, "q-budget", "19800")
        await _set(pilot, app, "q-tpc", "completo")
        await _press(pilot, app, "q-add")
        item = schedule.load_state(qdir).items[0]
        assert (item.settings.method, item.settings.generations, item.settings.train_per_class) == ("mono", None, None)
        assert item.settings.population == 30 and item.settings.budget_gpu_s == 19800
        [row] = [[str(c) for c in app.query_one("#q-table", DataTable).get_row_at(0)]]
        assert row[4] == "mono" and row[6] == "30" and row[7] == "sem limite"  # method, population, generations
        # the suggested name follows the form: the method first (not for joint), then "g" + sl
        assert app.query_one("#q-name", Input).value.startswith("mono-fp32-p30-gsl-")

        # editing shows the same words back, and saving keeps them
        await _press(pilot, app, "q-edit")
        assert app.query_one("#q-method", Select).value == "mono"
        assert app.query_one("#q-generations", Input).value == "sem limite"
        assert app.query_one("#q-tpc", Input).value == "completo (45k)"
        app.query_one("#q-method", Select).value = "random"
        await _press(pilot, app, "q-save")
        saved = schedule.load_state(qdir).items[0].settings
        assert (saved.method, saved.generations, saved.train_per_class) == ("random", None, None)

        # "0" also means no limit; a number is a number; text is refused
        await _set(pilot, app, "q-name", "outra")
        await _set(pilot, app, "q-generations", "0")
        await _set(pilot, app, "q-tpc", "100")
        await _press(pilot, app, "q-add")
        assert schedule.load_state(qdir).items[1].settings.generations is None
        assert schedule.load_state(qdir).items[1].settings.train_per_class == 100
        await _set(pilot, app, "q-name", "ruim")
        await _set(pilot, app, "q-generations", "muitas")
        await _press(pilot, app, "q-add")
        assert "[erro]" in app.last_message and "inteiro" in app.last_message and len(_names(qdir)) == 2


async def test_invalid_value_shows_error_and_adds_nothing(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _set(pilot, app, "q-population", "999")
        await _press(pilot, app, "q-add")
        assert "[erro]" in app.last_message and "population" in app.last_message
        await _set(pilot, app, "q-population", "abc")
        await _press(pilot, app, "q-add")
        assert "[erro]" in app.last_message and "inteiro" in app.last_message
        assert _names(qdir) == []


async def test_edit_planned_item(tmp_path: Path, qdir: Path) -> None:
    _seed(qdir, "alfa")
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        assert app.query_one("#q-save", Button).disabled
        await _press(pilot, app, "q-edit")
        assert app.query_one("#q-name", Input).value == "alfa"
        assert not app.query_one("#q-save", Button).disabled
        await _set(pilot, app, "q-name", "beta")
        await _set(pilot, app, "q-generations", "7")
        await _press(pilot, app, "q-save")
        item = schedule.load_state(qdir).items[0]
        assert (item.name, item.settings.generations) == ("beta", 7)
        assert app.query_one("#q-save", Button).disabled


async def test_running_item_cannot_be_edited_or_deleted(tmp_path: Path, qdir: Path) -> None:
    _seed(qdir, "alfa")
    schedule.mutate(qdir, lambda s: schedule.set_item_state(s, "0001", schedule.RUNNING, run_id="r1"))
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _press(pilot, app, "q-edit")
        assert "[erro]" in app.last_message and app.query_one(QueuePane).editing_id is None
        app.messages.clear()
        await _press(pilot, app, "q-del")
        assert "[erro]" in app.last_message
        assert _names(qdir) == ["alfa"]


async def test_move_up_and_down(tmp_path: Path, qdir: Path) -> None:
    _seed(qdir, "a", "b", "c")
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        table = app.query_one("#q-table", DataTable)
        table.move_cursor(row=2)
        await pilot.pause()
        await _press(pilot, app, "q-up")
        assert _names(qdir) == ["a", "c", "b"]
        assert table.cursor_row == 1  # the cursor follows the moved item
        await _press(pilot, app, "q-down")
        assert _names(qdir) == ["a", "b", "c"]
        assert table.cursor_row == 2


async def test_delete_planned_keeps_history(tmp_path: Path, qdir: Path) -> None:
    _seed(qdir, "feita", "p1", "p2")
    schedule.mutate(qdir, lambda s: schedule.set_item_state(s, "0001", schedule.DONE, run_id="r1"))
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _press(pilot, app, "q-delall")
        await pilot.pause(0.2)
        assert _names(qdir) == ["feita", "p1", "p2"]  # waiting for the confirmation
        await pilot.click("#confirm")
        await pilot.pause(0.2)
        assert _names(qdir) == ["feita"]


async def test_sleep_save_writes_config_with_until(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _set(pilot, app, "s-population-min", "3")
        await _set(pilot, app, "s-population-max", "8")
        await _set(pilot, app, "s-until", "08:00")
        await _set(pilot, app, "s-seed", "42")
        app.query_one("#s-fp16_block", Checkbox).value = True
        await pilot.pause()
        await _press(pilot, app, "sleep-on")
        sleep = schedule.load_state(qdir).sleep
        assert sleep.enabled and sleep.population == (3, 8) and sleep.seed == 42
        assert sleep.slices == ("fp32", "fp16_block")
        assert sleep.until is not None and sleep.until[11:16] == "08:00"
        await _press(pilot, app, "sleep-off")
        assert not schedule.load_state(qdir).sleep.enabled


async def test_sleep_invalid_input_is_refused(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _set(pilot, app, "s-until", "25:99")
        await _press(pilot, app, "sleep-on")
        assert "[erro]" in app.last_message
        assert not schedule.load_state(qdir).sleep.enabled


async def test_run_queue_spawns_executor_after_clear_stop(
    tmp_path: Path, qdir: Path, fake_scheduler: list[Any]
) -> None:
    _seed(qdir, "alfa")
    (qdir / schedule.STOP_FILE).write_text("x", encoding="utf-8")
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _press(pilot, app, "q-run")
        assert len(fake_scheduler) == 1 and fake_scheduler[0][2] is False  # flag already cleared at spawn time
        assert fake_scheduler[0][1]["extra_env"] == {"CUDA_VISIBLE_DEVICES": ""}
        assert app.launched[0].pid == 4242


async def test_run_queue_refuses_when_executor_alive(tmp_path: Path, qdir: Path, fake_scheduler: list[Any]) -> None:
    _seed(qdir, "alfa")
    schedule.write_executor(qdir, pid=99, item_id=None, message="vivo")
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _press(pilot, app, "q-run")
        assert fake_scheduler == [] and "ja esta vivo" in app.last_message


async def test_stop_all_creates_flag_and_disables_sleep(tmp_path: Path, qdir: Path) -> None:
    schedule.mutate(qdir, lambda s: dataclasses.replace(s, sleep=dataclasses.replace(s.sleep, enabled=True)))
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await _press(pilot, app, "q-stop")
        await pilot.pause(0.2)
        assert not schedule.stop_requested(qdir)  # needs the confirmation first
        await pilot.click("#confirm")
        await pilot.pause(0.2)
        assert schedule.stop_requested(qdir)
        assert not schedule.load_state(qdir).sleep.enabled


async def test_read_only_disables_writes(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        for button_id in (
            "q-add",
            "q-edit",
            "q-save",
            "q-up",
            "q-down",
            "q-del",
            "q-delall",
            "q-run",
            "q-stop",
            "sleep-on",
            "sleep-off",
        ):
            assert app.query_one(f"#{button_id}", Button).disabled, button_id


async def test_monitor_tab_and_bindings(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert tabs.active == "tab-queue"  # opened to plan runs
        tabs.active = "tab-queue"
        await pilot.pause()
        # typing in an Input never fires the single-letter bindings
        app.query_one("#q-name", Input).focus()
        before = list(app.messages)
        await pilot.press("p", "s", "c", "r", "y")
        await pilot.pause()
        assert app.query_one("#q-name", Input).value.endswith("pscry")
        assert app.messages == before and app.show_synthetic is False
        # outside an Input the Monitor keys are off in the Fila tab
        app.query_one("#q-table", DataTable).focus()
        await pilot.press("y")
        await pilot.pause()
        assert app.show_synthetic is False
        tabs.active = "tab-monitor"
        await pilot.pause()
        app.query_one("#jobs", DataTable).focus()
        await pilot.press("y")
        await pilot.pause()
        assert app.show_synthetic is True


async def test_feedback_is_shown_inside_the_queue_tab(tmp_path: Path, qdir: Path) -> None:
    app = _app(tmp_path, qdir)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        pane_msg = app.query_one("#q-msg", Static)
        app.query_one("#q-name", Input).value = "com espaco"
        app.query_one("#q-add", Button).press()
        await pilot.pause()
        assert "[erro]" in str(pane_msg.render()) and "nome" in str(pane_msg.render())
        app.query_one("#q-name", Input).value = "teste_fila_1"
        app.query_one("#q-add", Button).press()
        await pilot.pause()
        assert [i.name for i in schedule.load_state(qdir).items] == ["teste_fila_1"]
        assert "adicionada" in str(pane_msg.render())
