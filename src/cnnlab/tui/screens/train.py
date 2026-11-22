"""Tela Treinar: sidebar de parametros, area principal alimentada por evento.

Nada de treino sai daqui. Os botoes montam um `JobSpec` e o empurram para a
fila do app; quem executa e a worker thread (ver `docs/01-ARCHITECTURE.md`).
`apply_core_event` e o unico ponto que mexe em widget, e roda na UI thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    ContentSwitcher,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Sparkline,
    Static,
)

from cnnlab.config import (
    ClassicJob,
    DatasetConfig,
    GAConfig,
    GAJob,
    JobSpec,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from cnnlab.events import (
    EpochEnd,
    Event,
    GenerationDone,
    IndividualDone,
    RunCancelled,
    RunFailed,
    RunFinished,
    RunStarted,
)
from cnnlab.tui.widgets.forms import labeled, section

if TYPE_CHECKING:  # evita ciclo: app.py importa esta tela
    from cnnlab.tui.app import TrainingApp

GA_BUDGET_INPUTS = frozenset({"ga-population", "ga-generations", "ga-epochs-ind", "ga-elitism"})

METRICS: tuple[tuple[str, str], ...] = (
    ("train_loss", "m-train-loss"),
    ("val_loss", "m-val-loss"),
    ("val_acc", "m-val-acc"),
)

IDLE_LINE = "sem run em andamento"
RETRAIN_LINE = "retreinando melhor genoma"


class CoreEvent(Message):
    """Evento do core embrulhado para atravessar a fronteira de thread.

    Mora aqui, e nao em `app.py`, porque `app.py` importa as telas: definir do
    outro lado fecharia um ciclo de import.
    """

    def __init__(self, evt: Event) -> None:
        self.evt = evt
        super().__init__()


class TrainView(Horizontal):
    """Sidebar de configuracao + painel de execucao."""

    DEFAULT_CLASSES = "view"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._run_id: str | None = None
        self._unit_total = 1
        self._fitness: list[float] = []
        # True depois que a barra global encostou em 100%. Enquanto ela estiver
        # cheia e ainda chegarem epocas, o que roda e o retreino final.
        self._reached_full = False

    def compose(self) -> ComposeResult:
        # Vertical, nao VerticalScroll: so os campos rolam, os botoes ficam.
        with Vertical(classes="sidebar"):
            yield section("MODO", first=True)
            with RadioSet(id="mode"):
                yield RadioButton("Classico", value=True, id="mode-classic")
                yield RadioButton("Algoritmo genetico", id="mode-ga")

            with ContentSwitcher(initial="fields-classic", id="mode-fields"):
                yield from self._classic_fields()
                yield from self._ga_fields()

            yield section("DATASET")
            yield from labeled(
                "classes",
                Input(
                    value="",
                    placeholder="100",
                    type="integer",
                    id="ds-subset",
                    compact=True,
                ),
            )
            yield Static("Vazio usa as 100 classes.", classes="hint")

            yield section("SESSAO")
            with Horizontal(classes="buttons"):
                yield Button("Iniciar", id="btn-start", classes="-primary")
                yield Button("Parar", id="btn-stop")
                yield Button("Enfileirar", id="btn-queue")

            yield section("MODO SLEEP")
            yield Static("Cada sessao recebe uma seed diferente (seed + i).", classes="hint")
            yield from labeled("N sessoes", Input(value="3", type="integer", id="sleep-n", compact=True))
            with Horizontal(classes="buttons"):
                yield Button("Enfileirar N", id="btn-sleep")

        with VerticalScroll(classes="main"):
            with Vertical(id="exec", classes="card"):
                yield Static(IDLE_LINE, id="run-line", classes="run-line")
                yield Label("Progresso global", classes="bar-label")
                yield ProgressBar(total=1, show_eta=False, id="pb-global")
                with Vertical(id="unit-progress"):
                    yield Label("Treinos individuais", classes="bar-label")
                    yield ProgressBar(total=1, show_eta=False, id="pb-unit")
                yield Label("Epoca", classes="bar-label")
                yield ProgressBar(total=1, show_eta=False, id="pb-epoch")
                with Horizontal(classes="metrics"):
                    for name, widget_id in METRICS:
                        yield Static(f"{name}\n[b]—[/]", id=widget_id, classes="metric")

            with Vertical(id="fitness", classes="card"):
                yield Static("Melhor fitness por geracao", classes="panel-title")
                yield Sparkline([], summary_function=max, id="spark-fitness")

            yield RichLog(markup=True, highlight=False, id="log")

    def on_mount(self) -> None:
        self.query_one("#exec").border_title = "Execucao"
        self.query_one("#fitness").border_title = "Evolucao"
        self.query_one("#log", RichLog).border_title = "Eventos"
        unit_progress = self.query_one("#unit-progress")
        unit_progress.styles.height = "auto"
        unit_progress.display = False
        self._refresh_budget()
        self.log_line("[dim]pronto.[/] configure a sessao e clique Iniciar")

    # -- entrada de eventos (UI thread) -----------------------------------

    def apply_core_event(self, evt: Event) -> None:
        """Chamada pelo app, ja na UI thread. Unico ponto que mexe em widget."""
        if isinstance(evt, RunStarted):
            self._on_run_started(evt)
        elif isinstance(evt, EpochEnd):
            self._on_epoch_end(evt)
        elif isinstance(evt, IndividualDone):
            self.log_line(f"{_clock(evt.ts)} gen {evt.gen} · ind {evt.individual} · fitness [b]{evt.fitness:.4f}[/]")
        elif isinstance(evt, GenerationDone):
            self._on_generation_done(evt)
        elif isinstance(evt, RunFinished):
            self._on_terminal(
                f"{_clock(evt.ts)} run concluida · melhor [b]{evt.best_fitness:.4f}[/]"
                f" · {evt.total_trainings} treinos · {_duration(evt.elapsed_s)}",
                "concluida",
            )
        elif isinstance(evt, RunFailed):
            self._on_terminal(f"{_clock(evt.ts)} [b red]falhou[/] {escape(evt.error)}", "falhou")
        elif isinstance(evt, RunCancelled):
            self._on_terminal(
                f"{_clock(evt.ts)} [b yellow]cancelada[/] no treino {evt.at_unit}",
                "cancelada",
            )

    def log_line(self, line: str) -> None:
        self.query_one("#log", RichLog).write(line)

    # -- handlers de evento ------------------------------------------------

    def _on_run_started(self, evt: RunStarted) -> None:
        self._run_id = evt.run_id
        self._unit_total = max(evt.unit_total, 1)
        self._reached_full = False
        self._fitness = []
        self.query_one("#spark-fitness", Sparkline).data = []

        self.query_one("#pb-global", ProgressBar).update(total=self._unit_total, progress=0)
        self.query_one("#unit-progress").display = evt.kind == "ga"
        self.query_one("#pb-unit", ProgressBar).update(total=self._unit_total, progress=0)
        self.query_one("#pb-epoch", ProgressBar).update(total=max(evt.epochs_per_unit, 1), progress=0)
        self._set_metrics(None, None, None)
        self.log_line(
            f"{_clock(evt.ts)} run [b]{escape(evt.run_id)}[/] iniciada · "
            f"{evt.unit_total} treinos · {evt.epochs_per_unit} epocas cada"
        )

    def _on_epoch_end(self, evt: EpochEnd) -> None:
        epochs_total = max(evt.epochs_total, 1)
        self.query_one("#pb-epoch", ProgressBar).update(total=epochs_total, progress=evt.epoch)
        self.query_one("#pb-unit", ProgressBar).update(
            total=self._unit_total,
            progress=min(evt.unit_index + 1, self._unit_total),
        )

        if self._reached_full:
            # A barra global ja encostou no total e ainda chegam epocas: o que
            # roda agora e o retreino final, ocupando o ultimo slot.
            self._set_run_line(f"{RETRAIN_LINE} · epoca {evt.epoch}/{epochs_total}")
        else:
            progress = min(evt.unit_index + evt.epoch / epochs_total, float(self._unit_total))
            self.query_one("#pb-global", ProgressBar).update(progress=progress)
            self._reached_full = progress >= self._unit_total
            self._set_run_line(_position_line(evt))

        self._set_metrics(evt.train_loss, evt.val_loss, evt.val_acc)

    def _on_generation_done(self, evt: GenerationDone) -> None:
        self._fitness.append(evt.best_fitness)
        # Lista nova a cada vez: o Sparkline so redesenha quando `data` e
        # reatribuido, nao quando a lista existente muda.
        self.query_one("#spark-fitness", Sparkline).data = list(self._fitness)
        self.log_line(
            f"{_clock(evt.ts)} gen {evt.gen} concluida · melhor "
            f"[b]{evt.best_fitness:.4f}[/] · media {evt.mean_fitness:.4f} · "
            f"desvio {evt.std_fitness:.4f}"
        )

    def _on_terminal(self, line: str, label: str) -> None:
        self._run_id = None
        self._reached_full = False
        self._set_run_line(f"{IDLE_LINE} · ultima run {label}")
        self.log_line(line)

    # -- interacao ---------------------------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Alterna os campos entre classico e AG."""
        if event.control.id != "mode":
            return
        switcher = self.query_one("#mode-fields", ContentSwitcher)
        switcher.current = "fields-ga" if event.pressed.id == "mode-ga" else "fields-classic"

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in GA_BUDGET_INPUTS:
            self._refresh_budget()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Nenhum treino sai daqui: tudo vira job na fila."""
        event.stop()
        if event.button.id == "btn-start":
            self._enqueue([self._build_spec()], start=True)
        elif event.button.id == "btn-queue":
            self._enqueue([self._build_spec()], start=False)
        elif event.button.id == "btn-sleep":
            self._enqueue_sleep()
        elif event.button.id == "btn-stop":
            self._stop_current()

    def _enqueue(self, specs: list[JobSpec] | None, start: bool) -> None:
        if specs is None:
            return
        app = self._app()
        app.enqueue(specs)
        if start:
            app.ensure_runner()
        kinds = ", ".join(sorted({spec.kind for spec in specs}))
        self.log_line(f"[dim]{len(specs)}[/] job(s) [b]{kinds}[/] na fila · {app.queue.pending_count()} pendente(s)")

    def _enqueue_sleep(self) -> None:
        """N sessoes de AG, cada uma com `seed + i`."""
        count = self._int("sleep-n", 0)
        if count < 1:
            self.log_line("[b red]N sessoes[/] precisa ser >= 1")
            return

        specs: list[JobSpec] = []
        for offset in range(count):
            spec = self._build_spec(seed_offset=offset, force_ga=True)
            if spec is None:
                return
            specs.append(spec)
        self._enqueue(specs, start=True)

    def _stop_current(self) -> None:
        if self._run_id is None:
            # Setar `cancel_current` com a fila parada contaminaria o proximo
            # job: o runner so limpa a flag no fim de cada job.
            self.log_line("[dim]nada em execucao para parar[/]")
            return
        self._app().cancel_current()
        self.log_line("[b yellow]parada pedida[/] · corta em ate uma epoca")

    # -- montagem do JobSpec ----------------------------------------------

    def _build_spec(self, seed_offset: int = 0, force_ga: bool = False) -> JobSpec | None:
        """Monta o job a partir da sidebar. `None` quando os campos nao validam."""
        is_ga = force_ga or self._mode_is_ga()
        try:
            dataset = DatasetConfig(class_subset=self._opt_int("ds-subset"))
            if is_ga:
                return GAJob(
                    dataset=dataset,
                    ga=GAConfig(
                        population_size=self._int("ga-population", 8),
                        generations=self._int("ga-generations", 6),
                        epochs_per_individual=self._int("ga-epochs-ind", 3),
                        crossover_prob=self._float("ga-crossover", 0.6),
                        mutation_prob=self._float("ga-mutation", 0.3),
                        elitism=self._int("ga-elitism", 1),
                        tournament_size=self._int("ga-tournament", 3),
                        seed=self._int("ga-seed", 42) + seed_offset,
                    ),
                )
            return ClassicJob(
                dataset=dataset,
                train=TrainConfig(
                    model=ModelConfig(
                        base_filters=self._int("cl-filters", 32),
                        n_blocks=self._int("cl-blocks", 3),
                        dropout=self._float("cl-dropout", 0.25),
                    ),
                    optim=OptimConfig(
                        optimizer=self._select("cl-optimizer", "adam"),
                        lr=self._float("cl-lr", 1e-3),
                        weight_decay=self._float("cl-wd", 1e-4),
                        batch_size=self._int("cl-batch", 128),
                    ),
                    epochs=self._int("cl-epochs", 10),
                    seed=self._int("cl-seed", 42) + seed_offset,
                ),
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(part) for part in first["loc"]) or "campo"
            self.log_line(f"[b red]{escape(field)}[/] {escape(first['msg'])}")
            return None

    def _mode_is_ga(self) -> bool:
        return self.query_one("#mode-fields", ContentSwitcher).current == "fields-ga"

    # -- internos ---------------------------------------------------------

    def _app(self) -> TrainingApp:
        return self.app  # type: ignore[return-value]

    def _set_run_line(self, text: str) -> None:
        self.query_one("#run-line", Static).update(text)

    def _set_metrics(self, train_loss: float | None, val_loss: float | None, val_acc: float | None) -> None:
        values = (
            "—" if train_loss is None else f"{train_loss:.4f}",
            "—" if val_loss is None else f"{val_loss:.4f}",
            "—" if val_acc is None else f"{val_acc * 100:.1f}%",
        )
        for (name, widget_id), value in zip(METRICS, values):
            self.query_one(f"#{widget_id}", Static).update(f"{name}\n[b]{value}[/]")

    def _classic_fields(self) -> ComposeResult:
        with VerticalScroll(id="fields-classic"):
            yield from labeled("lr", Input(value="0.001", type="number", id="cl-lr", compact=True))
            yield from labeled(
                "batch_size",
                Input(value="128", type="integer", id="cl-batch", compact=True),
            )
            yield from labeled(
                "optimizer",
                Select(
                    [("adam", "adam"), ("sgd_momentum", "sgd_momentum")],
                    value="adam",
                    allow_blank=False,
                    id="cl-optimizer",
                    compact=True,
                ),
            )
            yield from labeled(
                "dropout",
                Input(value="0.25", type="number", id="cl-dropout", compact=True),
            )
            yield from labeled(
                "weight_decay",
                Input(value="0.0001", type="number", id="cl-wd", compact=True),
            )
            yield from labeled(
                "base_filters",
                Input(value="32", type="integer", id="cl-filters", compact=True),
            )
            yield from labeled(
                "n_blocks",
                Input(value="3", type="integer", id="cl-blocks", compact=True),
            )
            yield from labeled("epochs", Input(value="10", type="integer", id="cl-epochs", compact=True))
            yield from labeled("seed", Input(value="42", type="integer", id="cl-seed", compact=True))

    def _ga_fields(self) -> ComposeResult:
        with VerticalScroll(id="fields-ga"):
            yield from labeled(
                "populacao",
                Input(value="8", type="integer", id="ga-population", compact=True),
            )
            yield from labeled(
                "geracoes",
                Input(value="6", type="integer", id="ga-generations", compact=True),
            )
            yield from labeled(
                "epocas/ind",
                Input(value="3", type="integer", id="ga-epochs-ind", compact=True),
            )
            yield from labeled(
                "crossover",
                Input(value="0.6", type="number", id="ga-crossover", compact=True),
            )
            yield from labeled(
                "mutacao",
                Input(value="0.3", type="number", id="ga-mutation", compact=True),
            )
            yield from labeled(
                "elitismo",
                Input(value="1", type="integer", id="ga-elitism", compact=True),
            )
            yield from labeled(
                "torneio",
                Input(value="3", type="integer", id="ga-tournament", compact=True),
            )
            yield from labeled("seed", Input(value="42", type="integer", id="ga-seed", compact=True))
            yield Static("", id="budget", classes="hint")

    def _refresh_budget(self) -> None:
        """Recalcula `N treinos · M epocas` a cada tecla nos campos do AG."""
        try:
            cfg = GAConfig(
                population_size=self._int("ga-population", 8),
                generations=self._int("ga-generations", 6),
                epochs_per_individual=self._int("ga-epochs-ind", 3),
                elitism=self._int("ga-elitism", 1),
            )
        except ValidationError:
            self.query_one("#budget", Static).update("orcamento: valores fora da faixa")
            return
        trainings, epochs = cfg.budget()
        self.query_one("#budget", Static).update(
            f"orcamento: [b]{trainings}[/] treinos · [b]{epochs}[/] epocas · +{cfg.final_epochs} no retreino final"
        )

    def _int(self, input_id: str, default: int) -> int:
        try:
            return int(self.query_one(f"#{input_id}", Input).value)
        except ValueError:
            return default

    def _opt_int(self, input_id: str) -> int | None:
        raw = self.query_one(f"#{input_id}", Input).value.strip()
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def _float(self, input_id: str, default: float) -> float:
        try:
            return float(self.query_one(f"#{input_id}", Input).value)
        except ValueError:
            return default

    def _select(self, select_id: str, default: str) -> str:
        value = self.query_one(f"#{select_id}", Select).value
        return value if isinstance(value, str) else default


def _clock(ts: str) -> str:
    """`2026-09-01T23:14:03` -> `[dim]23:14:03[/]`."""
    return f"[dim]{ts[11:19] or ts}[/]"


def _duration(elapsed_s: float) -> str:
    total = int(elapsed_s)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60:02d}s"


def _position_line(evt: EpochEnd) -> str:
    """Linha de posicao da run.

    `gen` e `individual` nao trazem seus totais em `EpochEnd` e `events.py` e
    contrato travado, entao o total que aparece e o de treinos da run, que ja
    diz onde a busca esta.
    """
    parts = [f"treino {evt.unit_index + 1}/{evt.unit_total}"]
    if evt.gen is not None:
        parts.append(f"gen {evt.gen}")
    if evt.individual is not None:
        parts.append(f"ind {evt.individual}")
    parts.append(f"epoca {evt.epoch}/{evt.epochs_total}")
    return " · ".join(parts)
