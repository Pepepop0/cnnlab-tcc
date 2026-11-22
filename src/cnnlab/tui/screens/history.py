"""Tela Historico: tabela de runs lida do disco + detalhe da run selecionada.

A lista vem de `storage.reader.list_runs()`, que le so `status.json` e
`summary.json` — barato o bastante para reler a cada entrada na tela, sem
cache. Os `.jsonl` so sao abertos quando o usuario seleciona uma run, que e
exatamente o caso previsto em `docs/07-SPEC-storage.md`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ValidationError
from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    RadioButton,
    RadioSet,
    Sparkline,
    Static,
)

from cnnlab.config import ClassicJob, GAJob, JobSpec
from cnnlab.events import RunKind
from cnnlab.storage.reader import (
    RunSummary,
    list_runs,
    read_config,
    read_generations,
    read_progress,
)
from cnnlab.tui.widgets.forms import section

COLUMNS = ("data", "tipo", "status", "melhor fitness", "duracao", "nota")

STATUS_COLORS: dict[str, str] = {
    "done": "#66d19e",
    "running": "#4fd6be",
    "queued": "#7d8b9a",
    "failed": "#f06c75",
    "cancelled": "#e2b341",
}

FILTERS: dict[str, RunKind | None] = {
    "filter-all": None,
    "filter-classic": "classic",
    "filter-ga": "ga",
}


def _fmt_date(created_at: str) -> str:
    """ISO -> `01/09 23:14`. Mantem o original se nao parsear."""
    try:
        return datetime.fromisoformat(created_at).strftime("%d/%m %H:%M")
    except ValueError:
        return created_at or "—"


def _fmt_duration(elapsed_s: float | None) -> str:
    if elapsed_s is None:
        return "—"
    total = int(elapsed_s)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def _fmt_fitness(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _fmt_genome(genome: dict[str, Any]) -> str:
    """Genoma vira uma linha. Escapado: as chaves vem do disco."""
    if not genome:
        return "—"
    return escape(" · ".join(f"{key} {value}" for key, value in genome.items()))


def _fmt_config(spec: JobSpec) -> str:
    """Resumo de uma linha por bloco da config usada na run."""
    if isinstance(spec, GAJob):
        ga = spec.ga
        return (
            f"ga · pop {ga.population_size} · {ga.generations} geracoes · "
            f"{ga.epochs_per_individual} epocas/ind · elitismo {ga.elitism} · "
            f"torneio {ga.tournament_size} · cx {ga.crossover_prob} · "
            f"mut {ga.mutation_prob} · seed {ga.seed}\n"
            f"dataset {spec.dataset.name} · "
            f"classes {spec.dataset.class_subset or 100} · "
            f"amp {'on' if spec.amp else 'off'}"
        )
    if isinstance(spec, ClassicJob):
        train, optim, model = spec.train, spec.train.optim, spec.train.model
        return (
            f"classic · {train.epochs} epocas · lr {optim.lr} · "
            f"bs {optim.batch_size} · {optim.optimizer} · "
            f"wd {optim.weight_decay} · seed {train.seed}\n"
            f"modelo {model.base_filters}f · {model.n_blocks} blocos · "
            f"dropout {model.dropout} · amp {'on' if train.amp else 'off'}"
        )
    return "config em formato desconhecido"


class HistoryView(Horizontal):
    """Filtro na sidebar, tabela e detalhe na area principal."""

    DEFAULT_CLASSES = "view"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runs: list[RunSummary] = []
        self._kind: RunKind | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="sidebar"):
            yield section("FILTRO", first=True)
            with RadioSet(id="filter-kind"):
                yield RadioButton("Todos", value=True, id="filter-all")
                yield RadioButton("Classico", id="filter-classic")
                yield RadioButton("Algoritmo genetico", id="filter-ga")

            yield section("RESUMO")
            yield Static("", id="counts", classes="hint")

            yield section("ACOES")
            with Horizontal(classes="buttons"):
                yield Button("Recarregar", id="btn-reload")
            yield Static("A lista e relida do disco a cada entrada na tela.", classes="hint")

        with VerticalScroll(classes="main"):
            yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail", classes="card"):
                yield Static("", id="detail-run", classes="panel-title")
                yield Static("", id="detail-config", classes="hint")
                yield Static("", id="detail-curve-label", classes="bar-label")
                yield Sparkline([0.0], summary_function=max, id="detail-spark")
                yield Static("", id="detail-genome")

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.border_title = "Runs"
        table.add_columns(*COLUMNS)
        self.query_one("#detail").border_title = "Detalhe"
        self.reload()

    def on_show(self) -> None:
        """Toda entrada na tela relê o disco. Sem cache, como manda a spec."""
        self.reload()
        # Foco na tabela: sem isso as setas nao andam nas linhas.
        self.query_one("#runs", DataTable).focus()

    # -- carga -------------------------------------------------------------

    def reload(self) -> None:
        """Relê `runs/` e repopula a tabela. Só pode rodar na UI thread."""
        self._runs = list_runs(kind=self._kind)
        table = self.query_one("#runs", DataTable)
        table.clear()
        for run in self._runs:
            table.add_row(
                _fmt_date(run.created_at),
                run.kind,
                Text(run.status, style=STATUS_COLORS.get(run.status, "")),
                _fmt_fitness(run.best_fitness),
                _fmt_duration(run.elapsed_s),
                run.note or "—",
                key=run.run_id,
            )

        done = sum(1 for run in self._runs if run.status == "done")
        failed = sum(1 for run in self._runs if run.status == "failed")
        self.query_one("#counts", Static).update(f"{len(self._runs)} runs · {done} concluidas · {failed} com falha")
        self._show_detail(self._runs[0] if self._runs else None)

    # -- interacao ---------------------------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Filtra por tipo relendo o disco: o filtro vai para `list_runs`."""
        if event.control.id != "filter-kind":
            return
        self._kind = FILTERS.get(event.pressed.id or "", None)
        self.reload()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        run_id = str(event.row_key.value)
        run = next((r for r in self._runs if r.run_id == run_id), None)
        if run is not None:
            self._show_detail(run)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-reload":
            self.reload()
        event.stop()

    # -- detalhe -----------------------------------------------------------

    def _show_detail(self, run: RunSummary | None) -> None:
        """Abre os arquivos da run selecionada. E aqui que os `.jsonl` sao lidos."""
        if run is None:
            self.query_one("#detail-run", Static).update("nenhuma run em runs/")
            self.query_one("#detail-config", Static).update("Rode um treino para popular o historico.")
            self.query_one("#detail-curve-label", Static).update("")
            self.query_one("#detail-genome", Static).update("")
            self.query_one("#detail-spark", Sparkline).data = [0.0]
            return

        header = f"{run.run_id} · {run.status}"
        if run.best_fitness is not None:
            header += f" · melhor {run.best_fitness:.4f}"
        self.query_one("#detail-run", Static).update(header)
        self.query_one("#detail-config", Static).update(self._config_text(run.run_id))

        curve, label = self._curve(run.run_id)
        self.query_one("#detail-curve-label", Static).update(label)
        self.query_one("#detail-spark", Sparkline).data = curve or [0.0]
        self.query_one("#detail-genome", Static).update(f"melhor genoma: {self._genome_text(run.run_id)}")

    def _config_text(self, run_id: str) -> str:
        try:
            return _fmt_config(read_config(run_id))
        except (OSError, ValueError, ValidationError):
            return "config.json ausente ou ilegivel"

    def _curve(self, run_id: str) -> tuple[list[float], str]:
        """Fitness por geracao no AG; na falta dele, val_acc por epoca."""
        generations = read_generations(run_id)
        curve = [float(row["best_fitness"]) for row in generations if isinstance(row.get("best_fitness"), (int, float))]
        if curve:
            unit = "geracao" if len(curve) == 1 else "geracoes"
            return curve, f"melhor fitness por geracao ({len(curve)} {unit})"

        progress = read_progress(run_id)
        curve = [float(row["val_acc"]) for row in progress if isinstance(row.get("val_acc"), (int, float))]
        if curve:
            unit = "epoca" if len(curve) == 1 else "epocas"
            return curve, f"val_acc por epoca ({len(curve)} {unit})"
        return [], "sem curva registrada"

    def _genome_text(self, run_id: str) -> str:
        generations = read_generations(run_id)
        for row in reversed(generations):
            genome = row.get("best_genome")
            if isinstance(genome, dict) and genome:
                return _fmt_genome(genome)
        return "—"
