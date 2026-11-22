"""Minimal Textual TUI to operate a pilot job of the TCC v2 demo.

    python -m cnnlab.tcc.demo_tui --db DB --artifacts DIR [--spec SPEC.json]
    python -m cnnlab.tcc.demo_tui --db DB --artifacts DIR --read-only     # only watch
    python -m cnnlab.tcc.demo_tui --db DB --artifacts DIR --queue-dir DIR # adds the "Fila" tab

Keys: `p` prepare, `s` start or resume, `c` cancel, `r` refresh, `y` show or
hide synthetic numbers, `Esc` back to the table, `q` quit.

The TUI is a thin controller over `operation.launcher`:

* with `--queue-dir` the monitor moves into a "Monitor" tab and a "Fila" tab
  (`demo_queue.QueuePane`) lets the operator plan runs, run them through the
  detached executor, use sleep mode and "Parar tudo". The keys above only act
  in the Monitor tab; typing in a text field never triggers them;
* it never imports the science front and never runs GPU work: `prepare` is a
  child process and `start` spawns a detached worker that keeps running when
  the TUI closes. Liveness comes from the lease in the database;
* status comes from the read model (read-only) on a timer, in a thread; no
  widget is touched outside the UI thread. Runs that no job refers to (every
  run of a schema v1 database) are listed too, with their results;
* with `--read-only` the keys that write are disabled. Without it, a database
  of an older schema is still never migrated from here: that is a deliberate
  step for the CLI, after a backup;
* there is no default wiring. A spec pointing to the synthetic test wiring is
  refused unless the TUI was opened with `--allow-synthetic`, and synthetic
  numbers are hidden until `y` is pressed, then shown labeled.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, RichLog, Static, TabbedContent, TabPane

from cnnlab.tcc.operation import launcher
from cnnlab.tcc.storage.errors import StorageError
from cnnlab.tcc.storage.readmodel import ReadModelError, per_block, version_text

HIDDEN = "oculto"
NO_JOB = "sem job"
NOT_RECORDED = "(nao registrado)"
ACTIVE_STATES = ("queued", "interrupted")
RESUMABLE_FINAL_STATES = ("cancelled", "failed")  # final jobs of a run that did not finish
TERMINAL_STATES = ("done", "failed", "cancelled")

JOB_HEADERS = ("job", "estado", "metodo", "avaliador", "GPU s / orcamento", "val/inv/int/curso", "lease", "hb")
LOG_LINES, LOG_WIDTH = 4, 118
EVALUATION_HEADERS = ("tentativa", "ger", "status", "E", "L ms", "V MB", "S MB", "gpu s", "precisao por bloco", "obs")

ERROR_HINT: dict[str, str] = {
    "not_found": "Banco ainda nao existe. Informe o spec e tecle p (preparar) para cria-lo.",
    "not_a_file": "O caminho de --db e uma pasta. Feche e abra a TUI apontando para o arquivo tcc.sqlite3.",
    "not_sqlite": "O arquivo de --db nao abre como SQLite. Confira o caminho e os arquivos -wal/-shm ao lado.",
    "not_tcc": "O arquivo de --db nao e um banco do TCC v2. Use o banco criado pelo comando prepare.",
    "schema_too_old": "O arquivo de --db nao tem o esquema do TCC v2 aplicado. Use um banco criado pelo RunStore.",
    "schema_too_new": "Banco mais novo que este codigo. Atualize o checkout antes de operar.",
}
SYNTHETIC_REFUSED = (
    "Fiacao SINTETICA recusada: o spec aponta para um avaliador falso de teste. Use a fabrica real no spec; "
    "para testar so a interface, abra a TUI com --allow-synthetic."
)
READ_ONLY_REFUSED = "Modo somente leitura (--read-only): esta tecla esta desativada. Nada foi alterado."
MIGRATION_REFUSED = (
    "Este banco tem esquema anterior ao deste codigo e ja guarda resultados. Operar por aqui o migraria. "
    "Faca backup e migre de proposito pela CLI, ou use outro --db. Nada foi alterado."
)


def _cell(value: object) -> Text:
    return Text(str(value))  # never parsed as markup (paths and ids may contain brackets)


def _number(value: float | None, pattern: str) -> str:
    return "-" if value is None else format(value, pattern)


def short_id(identifier: str, run_id: str) -> str:
    """Identifier without the run prefix, so a row fits the screen (the title names the run)."""
    prefix = f"{run_id}."
    return identifier[len(prefix) :] if identifier.startswith(prefix) else identifier


def entries_of(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows of the first table: every job, then every run that no job refers to."""
    rows = [{**job, "key": job["job_id"]} for job in status["jobs"]]
    for item in status.get("runs_without_job", ()):
        run = item["run"]
        rows.append(
            {
                "key": f"run:{run['run_id']}",
                "job_id": None,
                "run_id": run["run_id"],
                "state": NO_JOB,
                "options": {},
                "lease_owner": None,
                "lease_live": None,
                "lease_expires_at": None,
                "heartbeat_count": None,
                "claim_count": None,
                "cancel_requested": False,
                "cancel_reason": None,
                "final_detail": None,
                "recent_events": [],
                "run": run,
                "progress": item["progress"],
            }
        )
    return rows


def is_synthetic(entry: Mapping[str, Any]) -> bool:
    return bool(entry["options"].get("synthetic_evaluator") or entry["progress"]["synthetic_results"])


def job_cells(job: Mapping[str, Any], *, show_synthetic: bool) -> tuple[str, ...]:
    """One row of the job table; numbers of a synthetic run are hidden unless asked."""
    run, progress = job["run"], job["progress"]
    synthetic = is_synthetic(job)
    visible = show_synthetic or not synthetic
    attempts = progress["attempts"]
    counts = "/".join(str(attempts.get(k, 0)) for k in ("valid", "invalid", "interrupted", "running"))
    gpu = f"{progress['active_gpu_s']:.1f}" + (" (sint.)" if synthetic else "")
    lease = ""
    if job["state"] == "running":
        lease = f"{job['lease_owner']} " + ("vivo" if job["lease_live"] else "VENCIDO")
    return (
        job["job_id"] if job["job_id"] is not None else f"({NO_JOB}) run {job['run_id']}",
        job["state"] + (" (cancel. pedido)" if job["cancel_requested"] and job["state"] == "running" else ""),
        f"{run['method']}/b{run['seed_block']}",
        "SINTETICO (teste)" if synthetic else "real",
        f"{gpu if visible else HIDDEN} / {progress['budget_gpu_s']:.0f}",
        (counts + (" (sint.)" if synthetic else "")) if visible else HIDDEN,
        lease,
        "-" if job["heartbeat_count"] is None else str(job["heartbeat_count"]),
    )


def evaluation_cells(
    row: Mapping[str, Any],
    *,
    show_synthetic: bool,
    run_id: str = "",
    front_ids: frozenset[str] = frozenset(),
    genome: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """One row of the evaluation table; unmeasured objectives are `-`, never zero."""
    notes = []
    if row["synthetic"]:
        notes.append("SINTETICO")
    if row.get("preliminary"):
        notes.append("preliminar")
    if row.get("candidate_id") in front_ids:
        notes.append("frente")
    if row["cache_hit"]:
        notes.append(f"cache de {short_id(row['source_attempt_id'], run_id)}")
    if row["recovered"]:
        notes.append("recuperada")
    if row["failure_code"]:
        notes.append(f"{row['failure_stage']}/{row['failure_code']}")
    values = (
        _number(row["e"], ".4f"),
        _number(row["l_ms"], ".4f"),
        _number(row["v_mb"], ".3f"),
        _number(row["s_mb"], ".3f"),
        _number(row["gpu_seconds"], ".1f"),
    )
    if row["synthetic"]:
        values = tuple(f"{v} (sint.)" if v != "-" else v for v in values) if show_synthetic else (HIDDEN,) * 5
    generation = "-" if row.get("generation") is None else str(row["generation"])
    precision = per_block(genome or {}, "precision", missing=NOT_RECORDED)
    return (short_id(row["attempt_id"], run_id), generation, row["status"], *values, precision, ", ".join(notes))


def log_excerpt(lines: Sequence[str], *, max_lines: int = LOG_LINES, width: int = LOG_WIDTH) -> list[str]:
    """Last lines of a worker log for the screen: no launch header, one screen row each."""
    body = [line for line in lines if line.strip() and not line.startswith("# ")]
    return [line if len(line) <= width else line[: width - 3] + "..." for line in body[-max_lines:]]


def transition_message(job: Mapping[str, Any], previous: str) -> tuple[str, str]:
    """What to tell the operator when a job changed state between two reads."""
    state = job["state"]
    level = {"failed": "erro", "interrupted": "erro", "done": "ok", "running": "ok"}.get(state, "info")
    text = f"Job {job['job_id']}: {previous} -> {state}"
    if state == "running":
        text += f" (worker {job['lease_owner']})"
    elif state == "interrupted":
        text += " (lease vencido: o worker parou sem encerrar; tecle s para retomar)"
    elif job["final_detail"]:
        text += f" ({job['final_detail']})"
    return level, text


def job_detail(
    job: Mapping[str, Any] | None,
    *,
    fronts: Sequence[Mapping[str, Any]] = (),
    has_job_queue: bool = True,
    show_synthetic: bool = False,
) -> str:
    """Version, plan and search state of the selected row, only from what the database records."""
    if job is None:
        return "Nenhuma run selecionada."
    run, progress = job["run"], job["progress"]
    visible = show_synthetic or not is_synthetic(job)
    first = f"run {job['run_id']} | estagio {run['stage']} | plano {run['plan_id']}"
    if job["claim_count"] is not None:
        first += f" | claims {job['claim_count']}"
    if run["stage"] == "pilot":
        first += " | piloto exploratorio (nao confirmatorio)"
    lines = [f"sistema: {version_text(run, missing=NOT_RECORDED)}", first]
    search = f"busca: {progress['generations']} geracoes | {progress['checkpoints']} checkpoints"
    latest = progress["latest_checkpoint"]
    if latest is not None and visible:
        search += f" (ultimo: ger. {latest['generation']}, {latest['unique_evaluations']} unicas, {latest['active_gpu_seconds']:.1f} s GPU)"
    lines.append(f"{search} | {progress['front_snapshots']} frentes")
    if fronts:
        front = fronts[-1]
        hv = "nao calculado" if front["hypervolume"] is None else f"{front['hypervolume']:.6g}"
        members = ", ".join(short_id(c, job["run_id"]) for c in front["candidate_ids"])
        lines.append(f"frente mais recente ({len(front['candidate_ids'])} membros, HV {hv}): {members}")
    if job["cancel_requested"]:
        lines.append(f"cancelamento pedido: {job['cancel_reason']}")
    if job["final_detail"]:
        lines.append(f"detalhe: {job['final_detail']}")
    if job["job_id"] is None:
        reason = "banco sem fila de jobs (esquema v1)" if not has_job_queue else "nenhum job aponta para esta run"
        lines.append(f"fila: {reason}; sem job, lease nem heartbeat")
    else:
        lines.append(f"eventos: {' > '.join(e['kind'] for e in job['recent_events']) or '-'}")
    return "\n".join(lines)


class ConfirmScreen(ModalScreen[bool]):
    """Yes/no question before an action that starts a process."""

    BINDINGS = [Binding("escape", "back", "Voltar")]

    def __init__(self, title: str, body: str, confirm_label: str) -> None:
        super().__init__()
        self._title, self._body, self._confirm_label = title, body, confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, id="dialog-title", markup=False)
            yield Static(self._body, id="dialog-body", markup=False)
            with Horizontal(id="dialog-buttons"):
                yield Button(self._confirm_label, id="confirm", variant="primary")
                yield Button("Voltar", id="back")

    def on_mount(self) -> None:
        self.query_one("#back", Button).focus()  # the safe choice is the default

    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#back")
    def action_back(self) -> None:
        self.dismiss(False)


class CancelScreen(ModalScreen[str | None]):
    """Asks for the reason of a cancel; an empty reason is not accepted."""

    BINDINGS = [Binding("escape", "back", "Voltar")]

    def __init__(self, job_id: str, state: str) -> None:
        super().__init__()
        self._job_id, self._state = job_id, state

    def compose(self) -> ComposeResult:
        effect = (
            "O worker termina a tentativa em curso, confirma o resultado e encerra o job."
            if self._state == "running"
            else "O job sai da fila agora. A run pode continuar depois com um novo job."
        )
        with Vertical(id="dialog"):
            yield Label(f"Cancelar o job {self._job_id} ({self._state})?", id="dialog-title", markup=False)
            yield Static(
                effect + " O cancelamento fica registrado e nao pode ser desfeito.", id="dialog-body", markup=False
            )
            yield Label("Motivo (obrigatorio):", id="reason-label")
            yield Input(placeholder="por que cancelar", id="reason")
            yield Static("", id="reason-error", markup=False)
            with Horizontal(id="dialog-buttons"):
                yield Button("Confirmar cancelamento", id="confirm", variant="error")
                yield Button("Voltar", id="back")

    def on_mount(self) -> None:
        self.query_one("#reason", Input).focus()

    @on(Input.Submitted, "#reason")
    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        reason = self.query_one("#reason", Input).value.strip()
        if not reason:
            self.query_one("#reason-error", Static).update("[erro] Informe o motivo do cancelamento.")
            return
        self.dismiss(reason)

    @on(Button.Pressed, "#back")
    def action_back(self) -> None:
        self.dismiss(None)


class DemoTui(App[None]):
    """Prepare, start, watch and cancel one pilot job."""

    TITLE = "TCC v2 - demo piloto"
    ENABLE_COMMAND_PALETTE = False
    SUB_TITLE = "operacao (piloto exploratorio)"
    CSS = """
    #banner, #message, #detail, #worker-log { padding: 0 1; }
    #message { height: auto; min-height: 1; }
    #spec-row { height: 3; }
    #spec-label { padding: 1 1 0 1; }
    #spec { width: 1fr; }
    .title { padding: 0 1; text-style: bold; }
    #jobs { height: 6; }
    #detail { height: auto; max-height: 8; }
    #evaluations { height: 1fr; min-height: 4; }
    #worker-log { height: auto; max-height: 6; }
    #history { height: 4; }
    ConfirmScreen, CancelScreen { align: center middle; }
    #dialog { width: 76; height: auto; border: solid $primary; padding: 1 2; background: $surface; }
    #dialog-title { text-style: bold; }
    #dialog-body { padding: 1 0; }
    #dialog-buttons { height: auto; padding-top: 1; }
    #dialog-buttons Button { margin-right: 2; }
    """
    BINDINGS = [
        Binding("p", "prepare", "Preparar"),
        Binding("s", "start", "Iniciar/retomar"),
        Binding("c", "cancel", "Cancelar job"),
        Binding("r", "refresh", "Atualizar"),
        Binding("y", "toggle_synthetic", "Sinteticos"),
        Binding("escape", "focus_jobs", "Tabela", show=False),
        Binding("q", "quit", "Sair"),
    ]

    def __init__(
        self,
        db_path: Path,
        artifact_root: Path,
        *,
        spec_path: Path | None = None,
        allow_synthetic: bool = False,
        read_only: bool = False,
        worker_id: str | None = None,
        lease_s: float = 120.0,
        heartbeat_s: float | None = None,
        refresh_s: float = 2.0,
        repo_root: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        queue_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.queue_dir = None if queue_dir is None else Path(queue_dir)
        self.db_path, self.artifact_root = Path(db_path), Path(artifact_root)
        self.initial_spec = "" if spec_path is None else str(spec_path)
        self.allow_synthetic = allow_synthetic
        self.read_only = read_only
        self.worker_id = worker_id or launcher.default_worker_id()
        self.lease_s, self.heartbeat_s, self.refresh_s = lease_s, heartbeat_s, refresh_s
        self.repo_root, self.extra_env = repo_root, extra_env
        self.status: dict[str, Any] | None = None
        self.entries: list[dict[str, Any]] = []
        self.read_error: ReadModelError | None = None
        self.selected_key: str | None = None
        self.show_synthetic = False
        self.busy = False
        self.launched: list[launcher.WorkerHandle] = []
        self.last_message = ""
        self.messages: list[str] = []
        self.known_states: dict[str, str] | None = None
        self.announce_next_read = False  # only the first good read after start or after an error is announced

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        if self.queue_dir is None:
            yield from self._compose_monitor()
        else:
            with TabbedContent(id="main-tabs", initial="tab-queue"):  # opened to plan runs
                with TabPane("Monitor", id="tab-monitor"):
                    yield from self._compose_monitor()
                with TabPane("Fila", id="tab-queue"):
                    from cnnlab.tcc.demo_queue import QueuePane

                    yield QueuePane(
                        self.queue_dir,
                        read_only=self.read_only,
                        repo_root=self.repo_root,
                        extra_env=self.extra_env,
                        say=self.say,
                        confirm=self._ask_confirm,
                        launched=self.launched,
                    )
        yield Footer()

    def _ask_confirm(self, title: str, body: str, label: str, on_yes: Any) -> None:
        """Confirmation dialog for the Fila tab (kept here so demo_queue never imports this module)."""
        self.push_screen(ConfirmScreen(title, body, label), lambda ok: on_yes() if ok else None)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """The monitor keys only act while the Monitor tab is showing (they are hidden in the Fila tab)."""
        if self.queue_dir is not None and action != "quit":
            try:
                if self.query_one("#main-tabs", TabbedContent).active != "tab-monitor":
                    return False
            except Exception:  # noqa: BLE001 - before compose / while closing
                return True
        return True

    def _compose_monitor(self) -> ComposeResult:
        yield Static("", id="banner", markup=False)
        yield Static("", id="message", markup=False)
        with Horizontal(id="spec-row"):
            yield Label("Spec da demo (JSON):", id="spec-label")
            yield Input(value=self.initial_spec, placeholder="caminho do arquivo de spec", id="spec")
        yield Static("JOBS E RUNS", classes="title")
        yield DataTable(id="jobs", cursor_type="row")
        yield Static("", id="detail", markup=False)
        yield Static("AVALIACOES", id="evaluations-title", classes="title", markup=False)
        yield DataTable(id="evaluations", cursor_type="row")
        yield Static("", id="worker-log", markup=False)
        yield RichLog(id="history", wrap=True, markup=False, max_lines=300)

    def on_mount(self) -> None:
        self.query_one("#jobs", DataTable).add_columns(*JOB_HEADERS)
        self.query_one("#evaluations", DataTable).add_columns(*EVALUATION_HEADERS)
        mode = " | --allow-synthetic ATIVO (teste da interface)" if self.allow_synthetic else ""
        mode += " | SOMENTE LEITURA" if self.read_only else ""
        self.query_one("#banner", Static).update(f"banco: {self.db_path} | artefatos: {self.artifact_root}{mode}")
        # Focusing a widget of another tab would switch to it: with a queue, start on the queue table.
        self.query_one("#jobs" if self.queue_dir is None else "#q-table", DataTable).focus()
        self.say("info", "Lendo o banco...")
        self.announce_next_read = True
        self.set_interval(self.refresh_s, self.action_refresh)
        self.action_refresh()

    # -- messages ----------------------------------------------------------

    def say(self, level: str, text: str) -> None:
        """Latest message on screen and in the history; the level is text, not only color."""
        self.last_message = f"[{level}] {text}"
        self.messages.append(self.last_message)
        self.announce_next_read = False  # a newer message (e.g. the result of a command) wins over the timer
        self.query_one("#message", Static).update(self.last_message)
        self.query_one("#history", RichLog).write(f"{datetime.now().strftime('%H:%M:%S')} {self.last_message}")

    def _from_thread(self, callback: Any, *args: Any) -> None:
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:  # the app is closing; nothing left to update
            pass

    # -- status ------------------------------------------------------------

    @property
    def selected(self) -> dict[str, Any] | None:
        return next((e for e in self.entries if e["key"] == self.selected_key), None)

    def action_refresh(self) -> None:
        self._read_status(self.selected_key)

    @work(thread=True, group="status", exit_on_error=False)
    def _read_status(self, key: str | None) -> None:
        log_lines: list[str] = []
        log_path = self.launched[-1].log_path if self.launched else launcher.latest_log(self.db_path)
        if log_path is not None:
            log_lines = [f"log do worker: {log_path}", *log_excerpt(launcher.tail(log_path))]
        try:
            status = launcher.read_status(self.db_path)
            entries = entries_of(status)
            chosen = next((e for e in entries if e["key"] == key), entries[-1] if entries else None)
            if chosen is not None:
                status = launcher.read_status(self.db_path, run_id=chosen["run_id"])
            self._from_thread(self._show_status, status, None if chosen is None else chosen["key"], log_lines)
        except ReadModelError as exc:
            self._from_thread(self._show_read_error, exc, log_lines)
        except (StorageError, OSError) as exc:
            self._from_thread(self._show_read_error, ReadModelError("read_failed", str(exc)), log_lines)

    def _show_read_error(self, exc: ReadModelError, log_lines: Sequence[str]) -> None:
        first = self.read_error is None or self.read_error.code != exc.code
        self.status, self.read_error, self.entries, self.selected_key = None, exc, [], None
        self.query_one("#jobs", DataTable).clear()
        self.query_one("#evaluations", DataTable).clear()
        self.query_one("#detail", Static).update("")
        self.query_one("#evaluations-title", Static).update("AVALIACOES")
        self.query_one("#worker-log", Static).update("\n".join(log_lines))
        if first and not self.busy:
            hint = ERROR_HINT.get(exc.code, "Confira o caminho de --db e tecle r para tentar de novo.")
            if exc.code == "not_found" and self.read_only:
                hint = "Banco nao encontrado. Confira o caminho de --db (modo somente leitura nao cria banco)."
            self.say("info" if exc.code == "not_found" else "erro", f"{hint} ({exc})")
            self.announce_next_read = True

    def _detail(self, status: Mapping[str, Any]) -> str:
        entry = self.selected
        fronts = status["fronts"] if entry is not None and status["run_id"] == entry["run_id"] else ()
        return job_detail(
            entry, fronts=fronts, has_job_queue=status.get("has_job_queue", True), show_synthetic=self.show_synthetic
        )

    def _show_status(self, status: dict[str, Any], chosen: str | None, log_lines: Sequence[str]) -> None:
        self.status, self.read_error = status, None
        self.entries = entries_of(status)
        keys = [e["key"] for e in self.entries]
        if self.selected_key not in keys:
            self.selected_key = chosen if chosen in keys else (keys[-1] if keys else None)

        jobs = self.query_one("#jobs", DataTable)
        jobs.clear()
        for entry in self.entries:
            jobs.add_row(*(_cell(c) for c in job_cells(entry, show_synthetic=self.show_synthetic)), key=entry["key"])
        if self.selected_key in keys:
            jobs.move_cursor(row=keys.index(self.selected_key), animate=False)

        self.query_one("#detail", Static).update(self._detail(status))
        self._show_evaluations(status)
        if log_lines:
            self.query_one("#worker-log", Static).update("\n".join(log_lines))
        previous, self.known_states = self.known_states, {j["job_id"]: j["state"] for j in status["jobs"]}
        for job in status["jobs"]:
            before = (previous or {}).get(job["job_id"])
            if before is not None and before != job["state"]:
                self.say(*transition_message(job, before))
        if self.announce_next_read and not self.busy:  # a command in progress reports its own result
            self._announce(status)
        if status["run_id"] is not None and self.selected is not None and status["run_id"] != self.selected["run_id"]:
            self.action_refresh()  # the selection changed while this read was on its way

    def _announce(self, status: Mapping[str, Any]) -> None:
        schema = f"{status['read_model']}, esquema v{status['schema_version']}"
        loose = len(status["runs_without_job"])
        if status["jobs"]:
            extra = f" e {loose} run(s) sem job" if loose else ""
            self.say("ok", f"{len(status['jobs'])} job(s){extra} no banco ({schema}).")
        elif loose:
            queue = "banco sem fila de jobs" if not status["has_job_queue"] else "nenhum job na fila"
            self.say("ok", f"{loose} run(s) no banco, {queue} ({schema}). Leitura apenas; nada foi alterado.")
        elif self.read_only:
            self.say("info", f"Banco sem runs ({schema}).")
        else:
            self.say("info", "Banco sem jobs. Informe o spec e tecle p (preparar), ou s para preparar e iniciar.")

    def _show_evaluations(self, status: Mapping[str, Any]) -> None:
        table = self.query_one("#evaluations", DataTable)
        table.clear()
        entry = self.selected
        current = entry is not None and status["run_id"] == entry["run_id"]
        rows = status["evaluations"] if current else []
        run_id = status["run_id"] or ""
        front_ids = frozenset(status["fronts"][-1]["candidate_ids"]) if current and status["fronts"] else frozenset()
        genomes = {g["candidate_id"]: g for g in status.get("genomes", ())} if current else {}
        for row in rows:
            cells = evaluation_cells(
                row,
                show_synthetic=self.show_synthetic,
                run_id=run_id,
                front_ids=front_ids,
                genome=genomes.get(row["candidate_id"]),
            )
            table.add_row(*(_cell(c) for c in cells), key=row["attempt_id"])
        if not rows:
            title = "AVALIACOES (nenhuma tentativa registrada para a linha selecionada)"
        else:
            valid = sum(1 for r in rows if r["status"] == "valid")
            title = f"AVALIACOES de {run_id}: {len(rows)} tentativas, {valid} validas"
        synthetic = sum(1 for r in rows if r["synthetic"])
        if synthetic and self.show_synthetic:
            title += f" - {synthetic} com VALORES SINTETICOS (teste), NAO SAO MEDICOES"
        elif synthetic:
            title += f" - {synthetic} sinteticas com numeros ocultos (y mostra, rotulados)"
        elif any(r.get("preliminary") for r in rows):
            title += " - medicoes reais, marcadas como preliminares pelo avaliador"
        self.query_one("#evaluations-title", Static).update(title)

    @on(DataTable.RowHighlighted, "#jobs")
    def _job_highlighted(self) -> None:
        # Read the cursor now: events queued while the table was refilled are stale.
        row = self.query_one("#jobs", DataTable).cursor_row
        if 0 <= row < len(self.entries) and self.entries[row]["key"] != self.selected_key:
            self.selected_key = self.entries[row]["key"]
            if self.status is not None:
                self.query_one("#detail", Static).update(self._detail(self.status))
            self.action_refresh()

    # -- actions -----------------------------------------------------------

    def action_focus_jobs(self) -> None:
        self.query_one("#jobs", DataTable).focus()

    def action_toggle_synthetic(self) -> None:
        self.show_synthetic = not self.show_synthetic
        self.say(
            "info",
            "Numeros sinteticos VISIVEIS, rotulados (sint.): sao de avaliador falso, nao sao medicoes."
            if self.show_synthetic
            else "Numeros sinteticos ocultos.",
        )
        if self.status is not None:
            self._show_status(self.status, self.selected_key, [])
        self.action_refresh()

    def _may_write(self) -> bool:
        """False, after saying why, when this session must not write to the database."""
        if self.read_only:
            self.say("info", READ_ONLY_REFUSED)
            return False
        try:
            launcher.ensure_current_schema(self.db_path)
        except launcher.MigrationNeededError as exc:
            self.say("erro", f"{MIGRATION_REFUSED} ({exc})")
            return False
        except (StorageError, OSError):
            pass  # unreadable path: the action itself reports what is wrong with it
        return True

    def _spec(self) -> tuple[Path, Any] | None:
        """Validated spec from the input, or None after telling the operator what to fix."""
        text = self.query_one("#spec", Input).value.strip()
        if not text:
            self.say("erro", "Informe o caminho do spec (JSON) no campo acima e tente de novo.")
            self.query_one("#spec", Input).focus()
            return None
        try:
            spec = launcher.load_spec(text)
        except ValueError as exc:
            self.say("erro", f"Spec invalido: {exc}. Corrija o arquivo ou o caminho e tente de novo.")
            return None
        if not spec.factory:
            self.say("erro", "O spec nao tem 'factory' (modulo:funcao). A TUI nao escolhe avaliador por conta propria.")
            return None
        if launcher.is_testing_factory(spec.factory) and not self.allow_synthetic:
            self.say("erro", SYNTHETIC_REFUSED)
            return None
        return Path(text), spec

    def action_prepare(self) -> None:
        if not self._may_write():
            return
        if self.busy:
            self.say("info", "Ja existe um preparo em andamento; aguarde.")
            return
        checked = self._spec()
        if checked is None:
            return
        self.busy = True
        self.say("info", f"Preparando run a partir de {checked[0]} (processo separado)...")
        self._prepare(checked[0])

    @work(thread=True, group="command", exit_on_error=False)
    def _prepare(self, spec_path: Path) -> None:
        try:
            result = launcher.run_prepare(
                self.db_path,
                self.artifact_root,
                spec_path,
                allow_synthetic=self.allow_synthetic,
                repo_root=self.repo_root,
                extra_env=self.extra_env,
            )
            self._from_thread(self._prepared, result.ok, result.message, result.warnings)
        except Exception as exc:  # timeout, missing interpreter...
            self._from_thread(self._prepared, False, f"{type(exc).__name__}: {exc}")

    def _prepared(self, ok: bool, message: str, warnings: Sequence[str] = ()) -> None:
        self.busy = False
        for warning in warnings:  # e.g. this spec created another run instead of continuing one
            self.say("aviso", warning)
        if ok:
            self.say("ok", f"{message}. Tecle s para iniciar o worker.")
        elif "SyntheticWiringError" in message:
            self.say("erro", f"{SYNTHETIC_REFUSED} ({message})")
        else:
            self.say("erro", f"Preparo recusado: {message}. Corrija e tecle p de novo.")
        self.action_refresh()

    def action_start(self) -> None:
        if not self._may_write():
            return
        jobs = self.status["jobs"] if self.status is not None else []
        running = next((j for j in jobs if j["state"] == "running" and j["lease_live"]), None)
        if running is not None:
            self.say(
                "erro",
                f"O job {running['job_id']} esta em execucao por {running['lease_owner']} (lease vivo). "
                "Um job de GPU por vez: aguarde terminar ou cancele com c.",
            )
            return
        job = self.selected
        if job is not None and job["job_id"] is not None and job["state"] in ACTIVE_STATES + ("running",):
            synthetic = bool(job["options"].get("synthetic_evaluator"))
            if synthetic and not self.allow_synthetic:
                self.say("erro", SYNTHETIC_REFUSED)
                return
            job_id = job["job_id"]
            body = (
                f"Retomar o job {job_id} (estado {job['state']}) da run {job['run_id']}.\n"
                f"Avaliador: {job['options'].get('evaluator')}" + (" - SINTETICO (teste)" if synthetic else "") + "\n"
                "O worker roda em processo separado e continua se a TUI fechar."
            )
            self.push_screen(
                ConfirmScreen("Iniciar worker?", body, "Iniciar worker"),
                lambda yes: self._spawn(job_id=job_id) if yes else self.say("info", "Inicio desistido."),
            )
            return
        if job is not None and job["job_id"] is not None and job["state"] in RESUMABLE_FINAL_STATES:
            run_id = job["run_id"]
            newer = [j for j in jobs if j["run_id"] == run_id and j["queue_seq"] > job["queue_seq"]]
            if not newer:  # the last job of its run: continue the run itself, by its id
                synthetic = bool(job["options"].get("synthetic_evaluator"))
                if synthetic and not self.allow_synthetic:
                    self.say("erro", SYNTHETIC_REFUSED)
                    return
                body = (
                    f"Continuar a run {run_id} com um job novo (o job {job['job_id']} terminou {job['state']}).\n"
                    "A run e a mesma: so o que falta e avaliado. Codigo diferente do gravado e recusado.\n"
                    f"Avaliador: {job['options'].get('evaluator')}"
                    + (" - SINTETICO (teste)" if synthetic else "")
                    + "\n"
                    "O worker roda em processo separado e continua se a TUI fechar."
                )
                self.push_screen(
                    ConfirmScreen("Continuar a run?", body, "Iniciar worker"),
                    lambda yes: self._spawn(run_id=run_id) if yes else self.say("info", "Inicio desistido."),
                )
                return
        checked = self._spec()
        if checked is None:
            return
        spec_path, spec = checked
        body = (
            f"Preparar (se preciso) e iniciar a run do spec {spec_path}.\n"
            f"Metodo {spec.method}, bloco {spec.seed_block}, orcamento {spec.budget_gpu_s:.0f} s de GPU.\n"
            f"Fabrica: {spec.factory}"
            + (" - SINTETICA (teste)" if launcher.is_testing_factory(spec.factory) else "")
            + "\n"
            "O worker roda em processo separado e continua se a TUI fechar."
        )
        self.push_screen(
            ConfirmScreen("Iniciar worker?", body, "Iniciar worker"),
            lambda yes: self._spawn(spec_path=spec_path) if yes else self.say("info", "Inicio desistido."),
        )

    def _spawn(self, *, spec_path: Path | None = None, job_id: str | None = None, run_id: str | None = None) -> None:
        try:
            handle = launcher.spawn_worker(
                self.db_path,
                self.artifact_root,
                spec_path=spec_path,
                job_id=job_id,
                run_id=run_id,
                worker_id=self.worker_id,
                lease_s=self.lease_s,
                heartbeat_s=self.heartbeat_s,
                allow_synthetic=self.allow_synthetic,
                repo_root=self.repo_root,
                extra_env=self.extra_env,
            )
        except (StorageError, OSError, ValueError) as exc:
            self.say("erro", f"Nao foi possivel iniciar o worker: {exc}")
            return
        self.launched.append(handle)
        self.say("ok", f"Worker iniciado (pid {handle.pid}). Log: {handle.log_path}")
        self.action_refresh()

    def action_cancel(self) -> None:
        if not self._may_write():
            return
        job = self.selected
        if job is None:
            self.say("erro", "Nenhum job selecionado para cancelar.")
            return
        if job["job_id"] is None:
            self.say("info", f"A run {job['run_id']} nao tem job na fila; nada a cancelar.")
            return
        if job["state"] in TERMINAL_STATES:
            self.say("info", f"O job {job['job_id']} ja terminou ({job['state']}); nada a cancelar.")
            return
        if job["cancel_requested"]:
            self.say("info", f"O cancelamento do job {job['job_id']} ja foi pedido; aguardando o worker encerrar.")
            return
        job_id = job["job_id"]
        self.push_screen(
            CancelScreen(job_id, job["state"]),
            lambda reason: self._cancel(job_id, reason) if reason else self.say("info", "Cancelamento desistido."),
        )

    @work(thread=True, group="command", exit_on_error=False)
    def _cancel(self, job_id: str, reason: str) -> None:
        try:
            job = launcher.request_cancel(self.db_path, self.artifact_root, job_id, reason=reason)
        except (StorageError, OSError) as exc:
            self._from_thread(self.say, "erro", f"Cancelamento falhou: {type(exc).__name__}: {exc}")
            return
        if job.state == "cancelled":
            text = f"Job {job_id} cancelado."
        else:
            text = f"Cancelamento do job {job_id} pedido; o worker termina a tentativa em curso e encerra."
        self._from_thread(self.say, "ok", text)
        self._from_thread(self.action_refresh)


# -- entry point ------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cnnlab.tcc.demo_tui", description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, required=True, help="SQLite database of the demo (created by prepare)")
    parser.add_argument("--artifacts", type=Path, required=True, help="artifact root (the database lives outside it)")
    parser.add_argument("--spec", type=Path, default=None, help="demo spec (JSON) with the wiring factory")
    parser.add_argument("--worker-id", default=f"tui-{platform.node() or 'host'}")
    parser.add_argument("--lease-s", type=float, default=120.0)
    parser.add_argument("--heartbeat-s", type=float, default=None)
    parser.add_argument("--refresh-s", type=float, default=2.0)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="queue directory (fila.json, executor); adds the Fila tab to plan and run queued runs",
    )
    parser.add_argument("--read-only", action="store_true", help="only watch: prepare, start and cancel are disabled")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="accept the synthetic test wiring (to test the interface only; results are fake and labeled)",
    )
    return parser


def build_app(argv: Sequence[str] | None = None) -> DemoTui:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts, db = args.artifacts.resolve(), args.db.resolve()
    if artifacts == db.parent or artifacts in db.parents:
        parser.error("the database must live outside the artifact root (--db inside --artifacts)")
    if args.refresh_s <= 0:
        parser.error("--refresh-s must be positive")
    return DemoTui(
        args.db,
        args.artifacts,
        spec_path=args.spec,
        allow_synthetic=args.allow_synthetic,
        read_only=args.read_only,
        worker_id=args.worker_id,
        lease_s=args.lease_s,
        heartbeat_s=args.heartbeat_s,
        refresh_s=args.refresh_s,
        repo_root=args.repo_root,
        queue_dir=args.queue_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    app = build_app(argv)
    app.run()
    alive = [w for w in app.launched if w.alive]
    for worker in alive:
        print(f"worker pid {worker.pid} segue em execucao; log: {worker.log_path}")
    if alive:
        print(f"acompanhe com: {sys.executable} -m cnnlab.tcc.operation.demo status --db {app.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
