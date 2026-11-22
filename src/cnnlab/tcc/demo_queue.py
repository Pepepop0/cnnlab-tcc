"""Queue tab of the demo TUI: plan runs, leave them queued, sleep mode and "stop all".

The widget is a thin controller over ``operation.schedule`` (the queue file
``fila.json`` and its rules) and ``operation.scheduler`` (the detached executor
process). It never runs GPU work: "Rodar fila" only spawns the executor.

Reads and writes of the small queue files happen on the UI thread (they are
local JSON files under a short lock), so no widget is touched from another
thread. The table is refreshed on a timer and keeps the cursor on the same item.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static, TabbedContent, TabPane

from cnnlab.tcc.operation import launcher, schedule
from cnnlab.tcc.operation.schedule import LIMITS, ScheduleError, SleepConfig

QUEUE_HEADERS = (
    "#",
    "nome",
    "tag",
    "estado",
    "metodo",
    "recorte",
    "pop",
    "ger",
    "orcamento s",
    "run",
    "detalhe",
)
METHOD_CHOICES = (
    ("joint", "NSGA-II conjunto (joint)"),
    ("mono", "Monoobjetivo, so E (mono)"),
    ("random", "Aleatoria (random)"),
    ("sequential", "Sequencial: arquitetura, depois compressao"),
)
# Words that mean "no value" in the two optional fields (generation limit, images per class).
NONE_WORDS = ("", "0", "sem limite", "sem", "completo", "completo (45k)", "nenhum", "none")
SHOWN_NONE = {"generations": "sem limite", "train_per_class": "completo (45k)"}
REFRESH_S = 2.0
# (input id suffix, label, RunSettings field)
RUN_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("population", "Populacao (par)", "population"),
    ("generations", "Geracoes", "generations"),
    ("budget", "Orcamento GPU (s)", "budget_gpu_s"),
    ("epochs", "Epocas", "epochs"),
    ("tpc", "Imagens por classe", "train_per_class"),
    ("search-seed", "Semente busca", "search_seed"),
    ("train-seed", "Semente treino", "train_seed"),
)
# (input id prefix, label, SleepConfig field)
SLEEP_RANGES: tuple[tuple[str, str, str], ...] = (
    ("population", "Populacao (par)", "population"),
    ("generations", "Geracoes", "generations"),
    ("epochs", "Epocas", "epochs"),
    ("tpc", "Imagens por classe", "train_per_class"),
)
WRITE_BUTTONS = ("q-add", "q-edit", "q-up", "q-down", "q-del", "q-delall", "q-run", "q-stop", "sleep-on", "sleep-off")

Confirm = Callable[[str, str, str, Callable[[], None]], None]  # title, body, confirm label, on yes


def _limits_hint(name: str) -> str:
    low, high = LIMITS[name]
    return f"{low}..{high}" + (f" ou {SHOWN_NONE[name]}" if name in SHOWN_NONE else "")


def _shown(value: Any, name: str) -> str:
    return SHOWN_NONE[name] if value is None else str(value)


def _opt_int_of(text: str, label: str) -> int | None:
    """An integer, or None for the words that mean "no generation limit" / "the full split"."""
    return None if text.strip().lower() in NONE_WORDS else _int_of(text, label)


def _int_of(text: str, label: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise ScheduleError(f"{label}: informe um numero inteiro (recebido {text.strip()!r})") from None


def queue_rows(state: schedule.QueueState) -> list[tuple[str, ...]]:
    """One table row per queue item, in queue order; the first cell of a row is not its key."""
    rows: list[tuple[str, ...]] = []
    for index, item in enumerate(state.items, start=1):
        s = item.settings
        rows.append(
            (
                str(index),
                item.name,
                item.tag,
                item.state,
                s.method,
                s.slice,
                str(s.population),
                _shown(s.generations, "generations"),
                str(s.budget_gpu_s),
                item.run_id or "-",
                item.detail or "",
            )
        )
    return rows


def status_line(state: schedule.QueueState, executor: Mapping[str, Any] | None, now: datetime | None = None) -> str:
    """Executor, current run and sleep mode in one line of text."""
    if executor is not None and executor.get("alive"):
        current = next((i for i in state.items if i.state == schedule.RUNNING), None)
        running = f"rodando {current.name}" if current else "ocioso"
        who = f"executor VIVO (pid {executor.get('pid')}, {running})"
    else:
        who = "executor PARADO"
    planned = sum(1 for i in state.items if i.editable)
    sleep = state.sleep
    if schedule.sleep_active(sleep, now):
        until = f"ate {sleep.until[11:16]}" if sleep.until else "ate o Parar tudo"
        mode = f"sono LIGADO {until}, {sleep.draws} sorteadas"
    elif sleep.enabled:
        mode = f"sono expirou ({sleep.draws} sorteadas)"
    else:
        mode = f"sono desligado ({sleep.draws} sorteadas)"
    return f"{who} | {planned} planejadas | {mode}"


class QueuePane(Horizontal):
    """Form (new run / sleep) on the left, the queue and its buttons on the right."""

    DEFAULT_CSS = """
    QueuePane { height: 1fr; }
    QueuePane #q-left { width: 58; }
    QueuePane #q-right { width: 1fr; padding-left: 1; }
    QueuePane .frow { height: 3; }
    QueuePane .flabel { width: 22; padding: 1 1 0 0; }
    QueuePane .frow Input { width: 1fr; }
    QueuePane .frow Select { width: 1fr; }
    QueuePane .range Input { width: 12; margin-right: 1; }
    QueuePane .buttons { height: auto; }
    QueuePane .buttons Button { margin-right: 1; min-width: 8; width: 1fr; }
    QueuePane .fscroll { height: 1fr; }
    QueuePane #q-table { height: 1fr; min-height: 5; }
    QueuePane #q-status { height: auto; }
    QueuePane #q-msg { height: auto; min-height: 1; padding: 0 0 1 0; }
    QueuePane #q-msg.erro { color: $error; text-style: bold; }
    QueuePane #q-msg.ok { color: $success; }
    QueuePane #q-big Button { width: 1fr; margin-right: 1; height: 3; }
    QueuePane .hint { color: $text-muted; padding: 0 0 1 0; }
    """

    def __init__(
        self,
        queue_dir: Path,
        *,
        read_only: bool = False,
        repo_root: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        say: Callable[[str, str], None] | None = None,
        confirm: Confirm | None = None,
        launched: list[launcher.WorkerHandle] | None = None,
        refresh_s: float = REFRESH_S,
    ) -> None:
        super().__init__()
        self.queue_dir = Path(queue_dir)
        self.read_only = read_only
        self.repo_root, self.extra_env = repo_root, extra_env
        self._say = say
        self._confirm = confirm
        self.launched = launched if launched is not None else []
        self.refresh_s = refresh_s
        self.editing_id: str | None = None
        self.state = schedule.QueueState()
        self._rows: list[tuple[str, ...]] = []
        self._ids: list[str] = []

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="q-left"):
            with TabbedContent(id="q-tabs"):
                # Fields scroll; the buttons stay pinned below them on short terminals.
                with TabPane("Nova run", id="q-tab-new"):
                    with VerticalScroll(classes="fscroll"):
                        yield from self._compose_new()
                    with Horizontal(classes="buttons"):
                        yield Button("Adicionar", id="q-add", variant="primary")
                        yield Button("Salvar edicao", id="q-save")
                        yield Button("Limpar", id="q-clear")
                with TabPane("Sono", id="q-tab-sleep"):
                    with VerticalScroll(classes="fscroll"):
                        yield from self._compose_sleep()
                    with Horizontal(classes="buttons"):
                        yield Button("Salvar e ligar sono", id="sleep-on", variant="primary")
                        yield Button("Desligar sono", id="sleep-off")
        with Vertical(id="q-right"):
            yield Static("", id="q-status", markup=False)
            yield Static("", id="q-msg", markup=False)  # feedback here: the app message line is in Monitor
            yield DataTable(id="q-table", cursor_type="row")
            with Horizontal(classes="buttons"):
                yield Button("Subir", id="q-up")
                yield Button("Descer", id="q-down")
                yield Button("Editar", id="q-edit")
                yield Button("Apagar", id="q-del")
                yield Button("Apagar todas", id="q-delall")
            with Horizontal(id="q-big", classes="buttons"):
                yield Button("Rodar fila", id="q-run", variant="success")
                yield Button("Parar tudo", id="q-stop", variant="error")

    def _compose_new(self) -> ComposeResult:
        defaults = schedule.RunSettings()
        with Horizontal(classes="frow"):
            yield Label("Metodo", classes="flabel")
            yield Select(
                [(label, value) for value, label in METHOD_CHOICES],
                value=defaults.method,
                allow_blank=False,
                id="q-method",
            )
        with Horizontal(classes="frow"):
            yield Label("Nome", classes="flabel")
            yield Input(placeholder="vazio = nome sugerido", id="q-name")
        with Horizontal(classes="frow"):
            yield Label("Recorte", classes="flabel")
            yield Select([(s, s) for s in schedule.SLICES], value=defaults.slice, allow_blank=False, id="q-slice")
        for suffix, label, name in RUN_FIELDS:
            with Horizontal(classes="frow"):
                yield Label(label, classes="flabel", markup=False)
                yield Input(
                    value=_shown(getattr(defaults, name), name) if name in SHOWN_NONE else str(getattr(defaults, name)),
                    placeholder=_limits_hint(name),
                    id=f"q-{suffix}",
                )

    def _compose_sleep(self) -> ComposeResult:
        sleep = SleepConfig()
        yield Label(
            "Sorteia runs (NSGA-II) dentro das faixas ate a hora ou o Parar tudo.", classes="hint", markup=False
        )
        for prefix, label, name in SLEEP_RANGES:
            low, high = getattr(sleep, name)
            with Horizontal(classes="frow range"):
                yield Label(f"{label} min/max", classes="flabel", markup=False)
                yield Input(value=str(low), placeholder=_limits_hint(name), id=f"s-{prefix}-min")
                yield Input(value=str(high), placeholder=_limits_hint(name), id=f"s-{prefix}-max")
        with Horizontal(classes="frow"):
            yield Label("Recortes", classes="flabel")
            yield Checkbox("fp32", value="fp32" in sleep.slices, id="s-fp32")
            yield Checkbox("fp16_block", value="fp16_block" in sleep.slices, id="s-fp16_block")
            yield Checkbox("fp16_regions", value="fp16_regions" in sleep.slices, id="s-fp16_regions")
        with Horizontal(classes="frow"):
            yield Label("Orcamento por run (s)", classes="flabel")
            yield Input(value=str(sleep.budget_gpu_s), placeholder=_limits_hint("budget_gpu_s"), id="s-budget")
        with Horizontal(classes="frow"):
            yield Label("Parar as (HH:MM)", classes="flabel")
            yield Input(placeholder="vazio = so no Parar tudo", id="s-until")
        with Horizontal(classes="frow"):
            yield Label("Semente", classes="flabel")
            yield Input(value=str(sleep.seed), placeholder="inteiro", id="s-seed")
        yield Static("", id="s-info", markup=False)

    def on_mount(self) -> None:
        self.query_one("#q-table", DataTable).add_columns(*QUEUE_HEADERS)
        if self.read_only:
            for button_id in WRITE_BUTTONS + ("q-save", "q-clear"):
                self.query_one(f"#{button_id}", Button).disabled = True
        else:
            self.query_one("#q-save", Button).disabled = True
        self.refresh_view()
        self._load_sleep_form()
        self.query_one("#q-name", Input).value = self.suggest_name()
        self.set_interval(self.refresh_s, self.refresh_view)

    # -- helpers -----------------------------------------------------------

    def say(self, level: str, text: str) -> None:
        msg = self.query_one("#q-msg", Static)
        msg.update(f"[{level}] {text}")
        msg.set_classes(level if level in ("erro", "ok") else "")
        if self._say is not None:
            self._say(level, text)

    def _load(self) -> schedule.QueueState | None:
        try:
            return schedule.load_state(self.queue_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.say("erro", f"fila ilegivel em {self.queue_dir}: {exc}")
            return None

    def _apply(self, change: Callable[[schedule.QueueState], schedule.QueueState]) -> bool:
        """Locked read-modify-write; True when it worked, otherwise the reason is on screen."""
        try:
            self.state = schedule.mutate(self.queue_dir, change)
        except ScheduleError as exc:
            self.say("erro", str(exc))
            return False
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.say("erro", f"nao consegui gravar a fila: {exc}")
            return False
        self.refresh_view()
        return True

    def suggest_name(self) -> str:
        """``fp32-p4-g2-HHMM`` from the form (``mono-`` / ``random-`` first for those methods), unique in the queue."""
        slice_value = self.query_one("#q-slice", Select).value
        method = self.query_one("#q-method", Select).value
        pop = self.query_one("#q-population", Input).value.strip() or "0"
        gen = self.query_one("#q-generations", Input).value.strip()
        gen = "sl" if gen.lower() in NONE_WORDS else gen  # "sl": no generation limit; the name cannot hold spaces
        prefix = "" if method == "joint" else f"{method}-"
        base = f"{prefix}{slice_value}-p{pop}-g{gen}-{datetime.now().strftime('%H%M')}".lower()
        taken = {i.name for i in self.state.items}
        name, n = base, 2
        while name in taken:
            name, n = f"{base}-{n}", n + 1
        return name

    def _form_settings(self) -> schedule.RunSettings:
        values = {
            name: (_opt_int_of if name in schedule.OPTIONAL_LIMITS else _int_of)(
                self.query_one(f"#q-{suffix}", Input).value, label
            )
            for suffix, label, name in RUN_FIELDS
        }
        slice_value = self.query_one("#q-slice", Select).value
        method = self.query_one("#q-method", Select).value
        return schedule.RunSettings(slice=str(slice_value), method=str(method), **values)

    def _form_name(self) -> str:
        return self.query_one("#q-name", Input).value.strip() or self.suggest_name()

    def selected_id(self) -> str | None:
        table = self.query_one("#q-table", DataTable)
        if not table.row_count or not 0 <= table.cursor_row < len(self._ids):
            return None
        return self._ids[table.cursor_row]

    def _selected_item(self) -> schedule.QueueItem | None:
        item_id = self.selected_id()
        return next((i for i in self.state.items if i.item_id == item_id), None)

    # -- refresh -----------------------------------------------------------

    def refresh_view(self) -> None:
        """Re-read the queue and the executor heartbeat; the cursor stays on the same item."""
        state = self._load()
        if state is None:
            return
        self.state = state
        table = self.query_one("#q-table", DataTable)
        keep = self.selected_id()
        rows = queue_rows(state)
        if rows != self._rows:
            self._rows, self._ids = rows, [i.item_id for i in state.items]
            table.clear()
            for item, row in zip(state.items, rows):
                table.add_row(*row, key=item.item_id)
            if keep in self._ids:
                table.move_cursor(row=self._ids.index(keep))
        executor = schedule.read_executor(self.queue_dir)
        text = status_line(state, executor)
        if schedule.stop_requested(self.queue_dir):
            text += " | PARAR pedido"
        self.query_one("#q-status", Static).update(text)
        self.query_one("#s-info", Static).update(f"sorteadas ate agora: {state.sleep.draws}")

    def _load_sleep_form(self) -> None:
        sleep = self.state.sleep
        for prefix, _label, name in SLEEP_RANGES:
            low, high = getattr(sleep, name)
            self.query_one(f"#s-{prefix}-min", Input).value = str(low)
            self.query_one(f"#s-{prefix}-max", Input).value = str(high)
        for s in schedule.SLEEP_SLICES:
            self.query_one(f"#s-{s}", Checkbox).value = s in sleep.slices
        self.query_one("#s-budget", Input).value = str(sleep.budget_gpu_s)
        self.query_one("#s-until", Input).value = sleep.until[11:16] if sleep.until else ""
        self.query_one("#s-seed", Input).value = str(sleep.seed)

    # -- new run form ------------------------------------------------------

    @on(Button.Pressed, "#q-add")
    def _add(self) -> None:
        try:
            settings, name = self._form_settings(), self._form_name()
        except ScheduleError as exc:
            self.say("erro", str(exc))
            return
        if self._apply(lambda s: schedule.add_item(s, name, settings)):
            self.say("info", f"run {name} adicionada a fila")
            self.query_one("#q-name", Input).value = self.suggest_name()

    @on(Button.Pressed, "#q-edit")
    def _edit(self) -> None:
        item = self._selected_item()
        if item is None:
            self.say("info", "Selecione uma run na fila para editar.")
            return
        if not item.editable:
            self.say("erro", f"{item.name} ja saiu da fila ({item.state}); so runs planejadas podem ser editadas")
            return
        if item.from_plan:
            self.say(
                "erro", f"{item.name} vem do plano congelado da campanha e nao se edita; apague-a e enfileire de novo"
            )
            return
        s = item.settings
        self.query_one("#q-name", Input).value = item.name
        self.query_one("#q-slice", Select).value = s.slice
        self.query_one("#q-method", Select).value = s.method
        for suffix, _label, name in RUN_FIELDS:
            self.query_one(f"#q-{suffix}", Input).value = (
                _shown(getattr(s, name), name) if name in SHOWN_NONE else str(getattr(s, name))
            )
        self.editing_id = item.item_id
        self.query_one("#q-save", Button).disabled = self.read_only
        self.query_one("#q-tabs", TabbedContent).active = "q-tab-new"
        self.say("info", f"Editando {item.name}: altere os campos e use Salvar edicao.")

    @on(Button.Pressed, "#q-save")
    def _save(self) -> None:
        if self.editing_id is None:
            self.say("info", "Nenhuma edicao em andamento: use Editar em uma run planejada.")
            return
        item_id = self.editing_id
        try:
            settings, name = self._form_settings(), self._form_name()
        except ScheduleError as exc:
            self.say("erro", str(exc))
            return
        if self._apply(lambda s: schedule.update_item(s, item_id, name, settings)):
            self.say("info", f"run {name} atualizada")
            self._end_edit()

    @on(Button.Pressed, "#q-clear")
    def _clear(self) -> None:
        defaults = schedule.RunSettings()
        self.query_one("#q-slice", Select).value = defaults.slice
        for suffix, _label, name in RUN_FIELDS:
            self.query_one(f"#q-{suffix}", Input).value = str(getattr(defaults, name))
        self._end_edit()
        self.query_one("#q-name", Input).value = self.suggest_name()

    def _end_edit(self) -> None:
        self.editing_id = None
        self.query_one("#q-save", Button).disabled = True

    # -- queue buttons -----------------------------------------------------

    def _move(self, delta: int) -> None:
        item_id = self.selected_id()
        if item_id is None:
            self.say("info", "Selecione uma run planejada.")
            return
        if self._apply(lambda s: schedule.move_item(s, item_id, delta)):
            table = self.query_one("#q-table", DataTable)
            if item_id in self._ids:
                table.move_cursor(row=self._ids.index(item_id))

    @on(Button.Pressed, "#q-up")
    def _up(self) -> None:
        self._move(-1)

    @on(Button.Pressed, "#q-down")
    def _down(self) -> None:
        self._move(1)

    @on(Button.Pressed, "#q-del")
    def _delete(self) -> None:
        item = self._selected_item()
        if item is None:
            self.say("info", "Selecione uma run planejada.")
            return
        if self._apply(lambda s: schedule.delete_item(s, item.item_id)):
            if self.editing_id == item.item_id:
                self._end_edit()
            self.say("info", f"run {item.name} apagada da fila")

    @on(Button.Pressed, "#q-delall")
    def _delete_all(self) -> None:
        planned = sum(1 for i in self.state.items if i.editable)
        if not planned:
            self.say("info", "Nao ha runs planejadas.")
            return

        def go() -> None:
            if self._apply(schedule.delete_planned):
                self._end_edit()
                self.say("info", f"{planned} runs planejadas apagadas; o historico foi mantido")

        self._ask(
            "Apagar planejadas",
            f"Apagar as {planned} runs planejadas? Runs ja iniciadas ficam no historico.",
            "Apagar",
            go,
        )

    def _ask(self, title: str, body: str, label: str, on_yes: Callable[[], None]) -> None:
        if self._confirm is None:
            on_yes()
        else:
            self._confirm(title, body, label, on_yes)

    # -- executor ----------------------------------------------------------

    @on(Button.Pressed, "#q-run")
    def _run_queue(self) -> None:
        executor = schedule.read_executor(self.queue_dir)
        if executor is not None and executor.get("alive"):
            self.say("info", f"O executor ja esta vivo (pid {executor.get('pid')}); ele segue a fila.")
            return
        state = self._load()
        if state is None:
            return
        if not any(i.editable for i in state.items) and not schedule.sleep_active(state.sleep):
            self.say("info", "Nada para rodar: adicione runs planejadas ou ligue o sono.")
            return
        try:
            from cnnlab.tcc.operation.scheduler import spawn_executor

            schedule.clear_stop(self.queue_dir)
            handle = spawn_executor(self.queue_dir, repo_root=self.repo_root, extra_env=self.extra_env)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.say("erro", f"nao consegui iniciar o executor: {exc}")
            return
        self.launched.append(handle)
        self.say("info", f"Executor iniciado (pid {handle.pid}). Ele segue a fila e continua se a TUI fechar.")
        self.refresh_view()

    @on(Button.Pressed, "#q-stop")
    def _stop_all(self) -> None:
        def go() -> None:
            try:
                schedule.request_stop(self.queue_dir)
            except (ScheduleError, OSError) as exc:
                self.say("erro", f"nao consegui pedir a parada: {exc}")
                return
            self.say("info", "Parar tudo pedido: sono desligado; o executor cancela a run atual e sai.")
            self.refresh_view()

        self._ask(
            "Parar tudo",
            "Desligar o sono, parar de sortear e cancelar a run em andamento? As planejadas continuam na fila.",
            "Parar tudo",
            go,
        )

    # -- sleep -------------------------------------------------------------

    def _sleep_from_form(self, enabled: bool) -> SleepConfig:
        ranges: dict[str, tuple[int, int]] = {}
        for prefix, label, name in SLEEP_RANGES:
            ranges[name] = (
                _int_of(self.query_one(f"#s-{prefix}-min", Input).value, f"{label} min"),
                _int_of(self.query_one(f"#s-{prefix}-max", Input).value, f"{label} max"),
            )
        slices = tuple(s for s in schedule.SLEEP_SLICES if self.query_one(f"#s-{s}", Checkbox).value)
        until_text = self.query_one("#s-until", Input).value.strip()
        until = schedule.next_occurrence(until_text) if until_text else None
        return SleepConfig(
            enabled=enabled,
            slices=slices,
            until=until,
            budget_gpu_s=_int_of(self.query_one("#s-budget", Input).value, "Orcamento por run"),
            seed=_int_of(self.query_one("#s-seed", Input).value, "Semente"),
            draws=self.state.sleep.draws,
            **ranges,
        )

    @on(Button.Pressed, "#sleep-on")
    def _sleep_on(self) -> None:
        try:
            config = self._sleep_from_form(True)
        except ScheduleError as exc:
            self.say("erro", str(exc))
            return
        # draws is read again inside the lock: the executor advances it while we edit the form
        if self._apply(lambda s: replace(s, sleep=replace(config, draws=s.sleep.draws))):
            until = f"ate {config.until[11:16]}" if config.until else "ate o Parar tudo"
            self.say("info", f"Sono ligado {until}. Use Rodar fila para o executor comecar a sortear.")

    @on(Button.Pressed, "#sleep-off")
    def _sleep_off(self) -> None:
        if self._apply(lambda s: replace(s, sleep=replace(s.sleep, enabled=False))):
            self.say("info", "Sono desligado; as runs ja sorteadas continuam na fila.")
