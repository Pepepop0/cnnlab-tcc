"""Run schedule of the demo: planned runs in a JSON file, outside the database.

The database never deletes a job (audit trail), so the queue the operator edits
lives here: a *planned* run can be added, edited, moved or deleted; once the
executor (``operation.scheduler``) picks it, it becomes a run and a job in the
shared database and only its state is mirrored back into this file.

Sleep mode draws new run settings from ranges and keeps the executor busy until
the operator presses "stop all" or the deadline passes. Draw ``n`` of a sleep
configuration is a pure function of ``(seed, n)``, so every drawn run can be
traced back to its draw.

Files in the queue directory (one directory per queue, next to ``db/``):

* ``fila.json``      -- :class:`QueueState`, written atomically under a lock file
* ``fila.lock``      -- lock held for each read-modify-write of ``fila.json``
* ``parar.flag``     -- "stop all" request: stop drawing, cancel the current job
* ``executor.json``  -- executor heartbeat: pid, current item, last beat
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

QUEUE_FILE = "fila.json"
LOCK_FILE = "fila.lock"
STOP_FILE = "parar.flag"
EXECUTOR_FILE = "executor.json"
DB_RELATIVE = Path("db") / "tcc.sqlite3"
ARTIFACTS_DIRNAME = "artifacts"
SPECS_DIRNAME = "specs"
WORK_DIRNAME = "work"

PLANNED, RUNNING, DONE, FAILED, CANCELLED = "planejada", "rodando", "concluida", "falhou", "cancelada"
STATES = (PLANNED, RUNNING, DONE, FAILED, CANCELLED)
FINAL_STATES = (DONE, FAILED, CANCELLED)
MANUAL, SLEEP, CAMPAIGN = "manual", "sono", "campanha"
# The slices the pilot wiring runs end to end, from the poorest to the richest in representation genes; ``admitted`` is the
# admitted space of the campaign. The sleep mode draws only from the first three.
SLICES = ("fp32", "fp16_block", "fp16_regions", "admitted")
SLEEP_SLICES = SLICES[:3]
# Methods a queue item can ask for. ``sequential`` needs a slice with a free representation gene to compress with
# and has no generation limit (its two stages are limited by GPU time).
METHODS = ("joint", "mono", "random", "sequential")
# Settings that take None: no generation limit (the GPU budget ends the run) / the full 45k train split.
OPTIONAL_LIMITS = ("generations", "train_per_class")
AUTHOR = "Pedro Antunes"
FACTORY = "cnnlab.tcc.demo_wiring:demo_wiring"
EXECUTOR_STALE_S = 30.0

# Inclusive limits of every setting; the TUI shows them next to the fields.
LIMITS: dict[str, tuple[int, int]] = {
    "population": (2, 64),
    "generations": (1, 50),
    "budget_gpu_s": (60, 86_400),
    "epochs": (1, 50),
    "train_per_class": (10, 450),  # the demo train split has 45k images, 450 per class
    "search_seed": (0, 2**31 - 1),
    "train_seed": (0, 2**31 - 1),
}
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


class ScheduleError(ValueError):
    """An edit the schedule refuses; the message is shown to the operator as is (Portuguese)."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    """What the operator chooses for one run (method ``joint`` unless another is asked for).

    ``generations=None`` runs without a generation limit and ``train_per_class=None`` trains on
    the full 45k split; both are campaign settings.
    """

    slice: str = "fp32"
    population: int = 4
    generations: int | None = 2
    budget_gpu_s: int = 3600
    epochs: int = 5
    train_per_class: int | None = 100
    search_seed: int = 1001
    train_seed: int = 2001
    method: str = "joint"
    # Set only by the campaign: a run of the frozen plan ``plan_id``, block ``seed_block``. Its seeds, budget
    # and arguments come from the plan, which the run checks them against; ``factory_extra`` holds the plan's
    # arguments that are not form fields (reference point, sequential settings).
    calibration_seed: int = 0
    seed_block: int = 0
    plan_id: str | None = None
    factory_extra: dict[str, Any] = field(default_factory=dict)
    code_label: str | None = None  # a campaign's runs share one label, hence one environment
    factory: str | None = None  # a campaign's wiring factory, as frozen in its plan (None: the demo's)

    def __post_init__(self) -> None:
        if self.slice not in SLICES:
            raise ScheduleError(f"recorte deve ser um de {SLICES}, nao {self.slice!r}")
        if self.method not in METHODS:
            raise ScheduleError(f"metodo deve ser um de {METHODS}, nao {self.method!r}")
        if self.method == "sequential":
            if self.generations is not None:
                raise ScheduleError(
                    "o metodo sequencial nao tem limite de geracoes (as duas etapas param pelo tempo de GPU): use sem limite"
                )
            if self.slice == "fp32":
                raise ScheduleError(
                    "o metodo sequencial precisa de um recorte com representacao livre para comprimir: use fp16_block, fp16_regions ou admitted"
                )
        for name, (low, high) in LIMITS.items():
            value = getattr(self, name)
            if value is None and name in OPTIONAL_LIMITS:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScheduleError(f"{name} deve ser inteiro")
            if not low <= value <= high:
                raise ScheduleError(f"{name} deve ficar entre {low} e {high} (recebido {value})")


@dataclass(frozen=True, slots=True)
class QueueItem:
    """One planned (or already started) run of the schedule."""

    item_id: str
    name: str
    tag: str
    settings: RunSettings
    state: str = PLANNED
    created_at: str = ""
    run_id: str | None = None
    job_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    detail: str = ""

    @property
    def editable(self) -> bool:
        return self.state == PLANNED

    @property
    def from_plan(self) -> bool:
        """A run of the frozen campaign plan: its settings and its place in the order are the plan's."""
        return self.settings.plan_id is not None


@dataclass(frozen=True, slots=True)
class SleepConfig:
    """Ranges sleep mode draws from (inclusive) and how long it keeps drawing."""

    enabled: bool = False
    population: tuple[int, int] = (4, 12)
    generations: tuple[int, int] = (3, 10)
    slices: tuple[str, ...] = ("fp32",)
    budget_gpu_s: int = 3600
    epochs: tuple[int, int] = (5, 5)
    train_per_class: tuple[int, int] = (100, 100)
    until: str | None = None  # local ISO datetime; None = until "stop all"
    seed: int = 0
    draws: int = 0  # how many runs were drawn so far; the next draw uses this index

    def __post_init__(self) -> None:
        for name in ("population", "generations", "epochs", "train_per_class"):
            low, high = getattr(self, name)
            lo_lim, hi_lim = LIMITS[name]
            if not lo_lim <= low <= high <= hi_lim:
                raise ScheduleError(f"faixa de {name} invalida: {low}..{high} (limites {lo_lim}..{hi_lim})")
        if not self.slices or any(s not in SLEEP_SLICES for s in self.slices):
            raise ScheduleError(f"recortes do sono devem vir de {SLEEP_SLICES}")
        low, high = LIMITS["budget_gpu_s"]
        if not low <= self.budget_gpu_s <= high:
            raise ScheduleError(f"budget_gpu_s deve ficar entre {low} e {high}")


@dataclass(frozen=True, slots=True)
class QueueState:
    """Everything in ``fila.json``."""

    items: tuple[QueueItem, ...] = ()
    sleep: SleepConfig = field(default_factory=SleepConfig)
    next_id: int = 1
    version: int = 1


# -- (de)serialization --------------------------------------------------------


def state_to_json(state: QueueState) -> dict[str, Any]:
    return asdict(state)


def state_from_json(data: dict[str, Any]) -> QueueState:
    sleep = dict(data.get("sleep") or {})
    for key in ("population", "generations", "epochs", "train_per_class", "slices"):
        if key in sleep:
            sleep[key] = tuple(sleep[key])
    items = tuple(QueueItem(**{**item, "settings": RunSettings(**item["settings"])}) for item in data.get("items", []))
    return QueueState(
        items=items,
        sleep=SleepConfig(**sleep),
        next_id=int(data.get("next_id", 1)),
        version=int(data.get("version", 1)),
    )


def queue_dir_paths(queue_dir: Path) -> dict[str, Path]:
    """Every path the queue uses, derived from its directory."""
    queue_dir = Path(queue_dir)
    return {
        "queue": queue_dir / QUEUE_FILE,
        "lock": queue_dir / LOCK_FILE,
        "stop": queue_dir / STOP_FILE,
        "executor": queue_dir / EXECUTOR_FILE,
        "db": queue_dir / DB_RELATIVE,
        "artifacts": queue_dir / ARTIFACTS_DIRNAME,
        "specs": queue_dir / SPECS_DIRNAME,
        "work": queue_dir / WORK_DIRNAME,
    }


def load_state(queue_dir: Path) -> QueueState:
    """Current schedule; an absent file is an empty schedule."""
    path = Path(queue_dir) / QUEUE_FILE
    if not path.exists():
        return QueueState()
    return state_from_json(json.loads(_read_text(path)))


def _read_text(path: Path, *, retry_s: float = 2.0) -> str:
    """Read a file another process may be replacing right now (Windows denies access meanwhile)."""
    deadline = time.monotonic() + retry_s
    while True:
        try:
            return path.read_text(encoding="utf-8-sig")
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _write_json_atomic(path: Path, data: Any, *, retry_s: float = 2.0) -> None:
    """Write via a unique temporary file and ``os.replace``.

    The temporary name is unique per call because two threads of one process
    (executor loop and heartbeat) write the same file. On Windows ``os.replace``
    fails while another process has the target open (the TUI reading it), so it
    is retried for a short while.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    deadline = time.monotonic() + retry_s
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


def save_state(queue_dir: Path, state: QueueState) -> None:
    _write_json_atomic(Path(queue_dir) / QUEUE_FILE, state_to_json(state))


@contextmanager
def locked(queue_dir: Path, *, timeout_s: float = 10.0, stale_s: float = 30.0) -> Iterator[None]:
    """Exclusive lock file for one read-modify-write; a lock older than ``stale_s`` is broken."""
    path = Path(queue_dir) / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_s:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise ScheduleError(f"fila ocupada por outro processo (trava {path}); tente de novo")
            time.sleep(0.05)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def mutate(queue_dir: Path, change: Callable[[QueueState], QueueState]) -> QueueState:
    """Load, apply ``change`` and save under the lock; returns the saved state."""
    with locked(queue_dir):
        state = change(load_state(queue_dir))
        save_state(queue_dir, state)
        return state


# -- editing (pure) -------------------------------------------------------------


def now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).replace(microsecond=0).isoformat()


def _check_even(settings: RunSettings) -> RunSettings:
    """NSGA-II crosses parents in pairs. Checked on add/edit only: history in the file must always load."""
    if settings.population % 2:
        raise ScheduleError(f"populacao deve ser par (o NSGA-II cruza em pares); recebido {settings.population}")
    return settings


def _even_sizes(bounds: tuple[int, int]) -> list[int]:
    """Even population sizes inside ``bounds``; a range without one (5..5) rounds up to the next even size."""
    sizes = [p for p in range(bounds[0], bounds[1] + 1) if p % 2 == 0]
    return sizes or [bounds[0] + 1]


def _check_name(state: QueueState, name: str, *, ignore: str | None = None) -> str:
    name = name.strip().lower()
    if not _NAME.match(name):
        raise ScheduleError(
            "nome: use letras minusculas, numeros, hifen ou _ (ate 48), sem espacos, comecando por letra ou numero"
        )
    if any(i.name == name and i.item_id != ignore for i in state.items):
        raise ScheduleError(f"ja existe uma run chamada {name!r} na fila")
    return name


def _find(state: QueueState, item_id: str) -> QueueItem:
    for item in state.items:
        if item.item_id == item_id:
            return item
    raise ScheduleError(f"item {item_id} nao esta na fila")


def _replace_item(state: QueueState, new: QueueItem) -> QueueState:
    return replace(state, items=tuple(new if i.item_id == new.item_id else i for i in state.items))


def add_item(
    state: QueueState, name: str, settings: RunSettings, *, tag: str = MANUAL, now: datetime | None = None
) -> QueueState:
    """Append a planned run at the end of the queue."""
    if tag not in (MANUAL, SLEEP, CAMPAIGN):
        raise ScheduleError(f"tag invalida: {tag!r}")
    item = QueueItem(
        item_id=f"{state.next_id:04d}",
        name=_check_name(state, name),
        tag=tag,
        settings=_check_even(settings),
        created_at=now_iso(now),
    )
    return replace(state, items=state.items + (item,), next_id=state.next_id + 1)


def update_item(state: QueueState, item_id: str, name: str, settings: RunSettings) -> QueueState:
    """Change a planned run; a run that already started is refused."""
    item = _find(state, item_id)
    if not item.editable:
        raise ScheduleError(f"{item.name} ja saiu da fila ({item.state}); so runs planejadas podem ser editadas")
    if item.from_plan:
        raise ScheduleError(
            f"{item.name} vem do plano congelado da campanha e nao se edita; apague-a e enfileire de novo"
        )
    return _replace_item(
        state, replace(item, name=_check_name(state, name, ignore=item_id), settings=_check_even(settings))
    )


def delete_item(state: QueueState, item_id: str) -> QueueState:
    """Remove a planned run; a run that already started stays (it is in the database)."""
    item = _find(state, item_id)
    if not item.editable:
        raise ScheduleError(f"{item.name} ja saiu da fila ({item.state}); nao pode ser apagada")
    return replace(state, items=tuple(i for i in state.items if i.item_id != item_id))


def delete_planned(state: QueueState) -> QueueState:
    """Remove every planned run; started and finished runs stay as history."""
    return replace(state, items=tuple(i for i in state.items if not i.editable))


def move_item(state: QueueState, item_id: str, delta: int) -> QueueState:
    """Move a planned run ``delta`` places among the planned runs (negative = earlier)."""
    item = _find(state, item_id)
    if not item.editable:
        raise ScheduleError(f"{item.name} ja saiu da fila; so runs planejadas mudam de ordem")
    if item.from_plan:
        raise ScheduleError(f"{item.name} segue a ordem do plano congelado da campanha; a ordem nao muda")
    planned = [i for i in state.items if i.editable]
    index = planned.index(item)
    target = max(0, min(len(planned) - 1, index + delta))
    planned.insert(target, planned.pop(index))
    order = iter(planned)
    return replace(state, items=tuple(next(order) if i.editable else i for i in state.items))


def next_planned(state: QueueState) -> QueueItem | None:
    """Runs the operator or the campaign put in the queue first, in queue order; then drawn (sleep) runs."""
    planned = [i for i in state.items if i.editable]
    return next((i for i in planned if i.tag != SLEEP), None) or next(iter(planned), None)


def set_item_state(
    state: QueueState,
    item_id: str,
    new_state: str,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    detail: str = "",
    now: datetime | None = None,
) -> QueueState:
    """Mirror what happened to a run; used by the executor only."""
    if new_state not in STATES:
        raise ScheduleError(f"estado invalido: {new_state!r}")
    item = _find(state, item_id)
    stamp = now_iso(now)
    changes: dict[str, Any] = {"state": new_state, "detail": detail}
    if run_id is not None:
        changes["run_id"] = run_id
    if job_id is not None:
        changes["job_id"] = job_id
    if new_state == RUNNING:
        changes["started_at"] = stamp
    if new_state in FINAL_STATES:
        changes["ended_at"] = stamp
    return _replace_item(state, replace(item, **changes))


# -- sleep mode -------------------------------------------------------------------


def next_occurrence(hhmm: str, now: datetime | None = None) -> str:
    """Local ISO datetime of the next ``HH:MM`` after ``now`` (tomorrow if already past)."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm.strip())
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        raise ScheduleError("horario deve ser HH:MM, por exemplo 08:00")
    now = now or datetime.now()
    target = now.replace(hour=int(match[1]), minute=int(match[2]), second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.isoformat()


def sleep_active(sleep: SleepConfig, now: datetime | None = None) -> bool:
    """Sleep mode is on and its deadline (if any) has not passed."""
    if not sleep.enabled:
        return False
    return sleep.until is None or (now or datetime.now()) < datetime.fromisoformat(sleep.until)


def draw_settings(sleep: SleepConfig, index: int) -> RunSettings:
    """Draw ``index`` of this sleep configuration; the same (seed, index) always gives the same settings."""
    rng = random.Random(f"cnnlab-tcc-sono:{sleep.seed}:{index}")
    return RunSettings(
        slice=rng.choice(sleep.slices),
        population=rng.choice(_even_sizes(sleep.population)),
        generations=rng.randint(*sleep.generations),
        budget_gpu_s=sleep.budget_gpu_s,
        epochs=rng.randint(*sleep.epochs),
        train_per_class=rng.randint(*sleep.train_per_class),
        search_seed=rng.randint(0, LIMITS["search_seed"][1]),
        train_seed=rng.randint(0, LIMITS["train_seed"][1]),
    )


def add_sleep_draw(state: QueueState, now: datetime | None = None) -> QueueState:
    """Append the next drawn run (tag ``sono``) and advance the draw counter."""
    index = state.sleep.draws
    name = f"sono-{(now or datetime.now()).strftime('%Y%m%d')}-s{state.sleep.seed}-{index + 1:03d}"
    added = add_item(state, name, draw_settings(state.sleep, index), tag=SLEEP, now=now)
    return replace(added, sleep=replace(state.sleep, draws=index + 1))


# -- spec -------------------------------------------------------------------------


def run_id_of(item: QueueItem) -> str:
    """Run id the executor asks for; readable in the page and unique inside the queue.

    A campaign run is named by its plan, method and block, not by its place in the queue, so the same
    cell has the same run id wherever it is queued.
    """
    if item.settings.plan_id is not None:
        s = item.settings
        return f"camp-{s.plan_id.rsplit('-', 1)[-1][:8]}-{s.method}-b{s.seed_block}"
    return f"fila-{item.item_id}-{item.name}"


def spec_payload(item: QueueItem) -> dict[str, Any]:
    """The ``DemoRunSpec`` mapping of one queue item (the method of the item; a campaign item is confirmatory)."""
    s = item.settings
    factory_args = {
        "train_per_class": s.train_per_class,
        "epochs": s.epochs,
        "population": s.population,
        "generations": s.generations,
        "slice": s.slice,
        **s.factory_extra,
    }
    payload = {
        "method": s.method,
        "seed_block": s.seed_block,
        "search_seed": s.search_seed,
        "train_seed": s.train_seed,
        "calibration_seed": s.calibration_seed,
        "budget_gpu_s": s.budget_gpu_s,
        "code_label": item.name,
        "author": AUTHOR,
        "run_id": run_id_of(item),
        "factory": s.factory or FACTORY,
        "factory_args": factory_args,
    }
    if s.plan_id is not None:
        payload.update(stage="confirmatory", plan_id=s.plan_id, code_label=s.code_label or item.name)
    return payload


def write_spec(queue_dir: Path, item: QueueItem) -> Path:
    """Write the spec of ``item`` under ``specs/`` (UTF-8 without BOM, atomic) and return its path."""
    path = queue_dir_paths(queue_dir)["specs"] / f"{item.item_id}-{item.name}.json"
    _write_json_atomic(path, spec_payload(item))
    return path


# -- stop flag and executor heartbeat -----------------------------------------------


def request_stop(queue_dir: Path) -> None:
    """ "Stop all": the executor stops drawing, cancels the current job and exits."""
    path = Path(queue_dir) / STOP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(now_iso(), encoding="utf-8")
    mutate(queue_dir, lambda s: replace(s, sleep=replace(s.sleep, enabled=False)))


def stop_requested(queue_dir: Path) -> bool:
    return (Path(queue_dir) / STOP_FILE).exists()


def clear_stop(queue_dir: Path) -> None:
    (Path(queue_dir) / STOP_FILE).unlink(missing_ok=True)


def write_executor(queue_dir: Path, *, pid: int, item_id: str | None, message: str) -> None:
    _write_json_atomic(
        Path(queue_dir) / EXECUTOR_FILE, {"pid": pid, "item_id": item_id, "message": message, "beat": time.time()}
    )


def read_executor(queue_dir: Path) -> dict[str, Any] | None:
    """Last heartbeat of the executor with ``alive`` (beat newer than ``EXECUTOR_STALE_S``), or None."""
    path = Path(queue_dir) / EXECUTOR_FILE
    try:
        data = json.loads(_read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    data["alive"] = time.time() - float(data.get("beat", 0)) < EXECUTOR_STALE_S
    return data
