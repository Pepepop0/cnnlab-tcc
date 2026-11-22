"""Tela Usar rede: carrega uma rede treinada e classifica uma imagem.

A inferencia roda numa worker thread (`@work(thread=True)`), nunca no handler
do botao: carregar `best.pt` e, no caso da amostra, subir o CIFAR-100 para a
VRAM sao operacoes de segundos. O que volta para a UI e uma dataclass imutavel
dentro de uma `Message` — nenhum tensor e nenhum widget cruzam a fronteira.

`predictions.jsonl` e escrito por `core/predict.py`, nao por aqui; esta tela so
o le para montar o historico.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    DataTable,
    Input,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from cnnlab.storage import layout
from cnnlab.storage.reader import artifact_path, list_runs, read_config
from cnnlab.tui.widgets.forms import labeled, section
from cnnlab.tui.widgets.thumbnail import Pixel, Thumbnail

BAR_WIDTH = 24
TOP5_SLOTS = 5
HISTORY_ROWS = 20
HISTORY_COLUMNS = ("hora", "imagem", "origem", "classe", "confianca")
SOURCE_LABELS = {"tensor": "amostra", "upload": "upload", "dataset": "amostra"}


@dataclass(frozen=True)
class PredictionOutcome:
    """Resultado pronto para a UI. Sem tensor, sem modelo, sem Path aberto."""

    run_id: str
    source: str
    caption: str
    predicted: str
    confidence: float
    top5: list[tuple[str, float]]
    num_classes: int
    truth: str | None = None
    pixels: list[list[Pixel]] = field(default_factory=list)


@dataclass(frozen=True)
class SampleOutcome:
    """Amostra preparada para preview, ainda sem carregar a rede."""

    run_id: str
    index: int
    truth: str
    caption: str
    pixels: list[list[Pixel]]


class PredictionDone(Message):
    def __init__(self, outcome: PredictionOutcome) -> None:
        self.outcome = outcome
        super().__init__()


class SampleReady(Message):
    def __init__(self, outcome: SampleOutcome) -> None:
        self.outcome = outcome
        super().__init__()


class PredictionFailed(Message):
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class PredictView(Horizontal):
    """Sidebar de entrada, area principal com resultado e historico."""

    DEFAULT_CLASSES = "view"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._run_ids: list[str] = []
        self._sample_run_id: str | None = None
        self._sample_index: int | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="sidebar"):
            yield section("REDE", first=True)
            yield from labeled(
                "run",
                Select(
                    [],
                    prompt="nenhuma run com best.pt",
                    id="run-select",
                    compact=True,
                ),
            )
            yield Static("", id="run-warning", classes="hint")

            yield section("IMAGEM")
            with RadioSet(id="source"):
                yield RadioButton("Amostra do dataset", value=True, id="src-dataset")
                yield RadioButton("Arquivo local", id="src-file")
            yield from labeled(
                "caminho",
                Input(placeholder="C:/fotos/gato.png", id="img-path", compact=True),
            )
            with Horizontal(classes="buttons"):
                yield Button("Escolher arquivo...", id="btn-file")
            yield Static(
                "O arquivo e copiado para uploads/ somente ao classificar.",
                classes="hint",
            )

            with Horizontal(classes="buttons"):
                yield Button("Classificar", id="btn-predict", classes="-primary")
                yield Button("Sortear", id="btn-sample")
            yield Static("", id="predict-status", classes="hint")

        with VerticalScroll(classes="main"):
            yield Thumbnail(id="preview")
            with Vertical(id="top5", classes="card"):
                for slot in range(TOP5_SLOTS):
                    with Horizontal(classes="top5-row"):
                        yield Static("—", id=f"top5-label-{slot}", classes="top5-label")
                        yield Static("", id=f"top5-bar-{slot}", classes="top5-bar")
            yield DataTable(id="predictions", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        preview = self.query_one("#preview", Thumbnail)
        preview.border_title = "Entrada"
        preview.show_message("sem imagem\nescolha um arquivo ou sorteie uma amostra")
        self.query_one("#top5").border_title = "Top-5"
        table = self.query_one("#predictions", DataTable)
        table.border_title = "Predicoes anteriores"
        table.add_columns(*HISTORY_COLUMNS)
        self.refresh_runs()
        self.refresh_history()

    def on_show(self) -> None:
        """Runs concluidas e historico sao relidos a cada entrada na tela."""
        self.refresh_runs()
        self.refresh_history()

    # -- carga -------------------------------------------------------------

    def refresh_runs(self) -> None:
        """So entram runs `done` que tenham `best.pt` no disco."""
        runs = [run for run in list_runs() if run.status == "done" and artifact_path(run.run_id) is not None]
        self._run_ids = [run.run_id for run in runs]
        options = [
            (
                f"{run.run_id}" + (f" · {run.best_fitness:.1%}" if run.best_fitness is not None else ""),
                run.run_id,
            )
            for run in runs
        ]

        select = self.query_one("#run-select", Select)
        previous = select.value
        select.set_options(options)
        if previous in self._run_ids:
            select.value = previous
        elif self._run_ids:
            select.value = self._run_ids[0]
        self._update_run_warning()

    def refresh_history(self) -> None:
        """Le `predictions.jsonl`, mais recentes primeiro."""
        table = self.query_one("#predictions", DataTable)
        table.clear()
        for row in reversed(_read_jsonl(layout.PREDICTIONS_FILE)[-HISTORY_ROWS:]):
            confidence = row.get("confidence")
            table.add_row(
                _fmt_time(str(row.get("ts", ""))),
                _fmt_image(row.get("image")),
                SOURCE_LABELS.get(str(row.get("source")), str(row.get("source", "—"))),
                str(row.get("predicted", "—")),
                f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—",
            )

    # -- interacao ---------------------------------------------------------

    @on(Select.Changed, "#run-select")
    def _on_run_changed(self, event: Select.Changed) -> None:
        if self._sample_index is not None:
            self._sample_run_id = None
            self._sample_index = None
            self.query_one("#preview", Thumbnail).show_message("a run mudou\nsorteie outra amostra")
            self._clear_result()
        self._update_run_warning()

    @on(Button.Pressed, "#btn-file")
    def _on_choose_file(self, event: Button.Pressed) -> None:
        event.stop()
        self._choose_local_file()

    @on(Button.Pressed, "#btn-predict")
    def _on_predict(self, event: Button.Pressed) -> None:
        event.stop()
        use_file = self.query_one("#src-file", RadioButton).value
        self._start(use_file=use_file)

    @on(Button.Pressed, "#btn-sample")
    def _on_sample(self, event: Button.Pressed) -> None:
        """Prepara outra amostra sem carregar nem executar a rede."""
        event.stop()
        self.query_one("#src-dataset", RadioButton).value = True
        self._start_sample()

    @on(SampleReady)
    def _on_sample_ready(self, message: SampleReady) -> None:
        self._set_busy(False)
        outcome = message.outcome
        selected_run = self.query_one("#run-select", Select).value
        if outcome.run_id != selected_run:
            return

        self._sample_run_id = outcome.run_id
        self._sample_index = outcome.index
        self.query_one("#preview", Thumbnail).show_pixels(outcome.pixels, outcome.caption)
        self._clear_result()
        self._status(
            f"amostra #{outcome.index} selecionada · clique Classificar",
            "ok",
        )

    @on(PredictionDone)
    def _on_done(self, message: PredictionDone) -> None:
        outcome = message.outcome
        self._set_busy(False)

        preview = self.query_one("#preview", Thumbnail)
        if outcome.pixels:
            preview.show_pixels(outcome.pixels, outcome.caption)
        else:
            preview.show_message(outcome.caption)

        for slot in range(TOP5_SLOTS):
            label = self.query_one(f"#top5-label-{slot}", Static)
            bar = self.query_one(f"#top5-bar-{slot}", Static)
            if slot < len(outcome.top5):
                name, probability = outcome.top5[slot]
                label.update(name)
                bar.update(_bar(probability))
            else:
                label.update("—")
                bar.update("")

        status = f"{outcome.predicted} · {outcome.confidence:.1%}"
        if outcome.truth is not None:
            hit = "acertou" if outcome.truth == outcome.predicted else "errou"
            status += f" · verdadeira {outcome.truth} ({hit})"
        self._status(status, "ok")
        self.refresh_history()

    @on(PredictionFailed)
    def _on_failed(self, message: PredictionFailed) -> None:
        self._set_busy(False)
        self._status(message.error, "err")

    # -- trabalho pesado ---------------------------------------------------

    def _start(self, *, use_file: bool) -> None:
        """Valida o que da para validar na UI e joga o resto para a thread."""
        run_id = self.query_one("#run-select", Select).value
        if not isinstance(run_id, str):
            self._status("selecione uma run concluida com best.pt", "warn")
            return

        raw_path = self.query_one("#img-path", Input).value.strip()
        if use_file and not raw_path:
            self._status("clique em Escolher arquivo", "warn")
            return
        if not use_file and (self._sample_run_id != run_id or self._sample_index is None):
            self._status("clique em Sortear antes de classificar", "warn")
            return

        self._set_busy(True)
        self._status(
            "carregando rede e imagem…" if use_file else "carregando rede e dataset…",
            "hint",
        )
        self._classify(run_id, use_file, raw_path, self._sample_index)

    def _start_sample(self) -> None:
        run_id = self.query_one("#run-select", Select).value
        if not isinstance(run_id, str):
            self._status("selecione uma run para usar o mesmo dataset", "warn")
            return

        self._set_busy(True)
        self._status("sorteando amostra do dataset...", "hint")
        self._sample(run_id)

    def _choose_local_file(self) -> None:
        try:
            selected = _open_image_dialog()
        except Exception as exc:
            self._status(f"seletor indisponivel: {type(exc).__name__}: {exc}", "err")
            return
        if not selected:
            self._status("nenhum arquivo selecionado", "hint")
            return

        path = Path(selected)
        try:
            pixels = _pixels_from_file(path)
        except Exception as exc:
            self._status(f"imagem invalida: {type(exc).__name__}: {exc}", "err")
            return

        self._sample_run_id = None
        self._sample_index = None
        self.query_one("#img-path", Input).value = str(path)
        self.query_one("#src-file", RadioButton).value = True
        self.query_one("#preview", Thumbnail).show_pixels(pixels, f"{path.name} · pronta para classificar")
        self._clear_result()
        self._status("arquivo selecionado · clique Classificar", "ok")

    @work(thread=True, exclusive=True, group="predict")
    def _sample(self, run_id: str) -> None:
        try:
            outcome = _load_dataset_sample(run_id)
        except Exception as exc:
            self.post_message(PredictionFailed(f"{type(exc).__name__}: {exc}"))
            return
        self.post_message(SampleReady(outcome))

    @work(thread=True, exclusive=True, group="predict")
    def _classify(
        self,
        run_id: str,
        use_file: bool,
        raw_path: str,
        sample_index: int | None,
    ) -> None:
        """Roda fora da UI thread. Nao toca em widget: so posta mensagem."""
        try:
            outcome = _run_prediction(run_id, use_file, raw_path, sample_index)
        except Exception as exc:  # a tela nunca pode morrer por imagem ruim
            self.post_message(PredictionFailed(f"{type(exc).__name__}: {exc}"))
            return
        self.post_message(PredictionDone(outcome))

    # -- internos ----------------------------------------------------------

    def _update_run_warning(self) -> None:
        """Avisa quando a run foi treinada num subconjunto de classes."""
        widget = self.query_one("#run-warning", Static)
        run_id = self.query_one("#run-select", Select).value
        if not isinstance(run_id, str):
            widget.update("")
            widget.set_class(False, "warn")
            return

        try:
            subset = read_config(run_id).dataset.class_subset
        except (OSError, ValueError):
            widget.update("config.json ilegivel: classes desconhecidas")
            widget.set_class(True, "warn")
            return

        if subset is None:
            widget.update("100 classes do CIFAR-100")
            widget.set_class(False, "warn")
        else:
            widget.update(
                f"treinada com class_subset={subset}: a rede so conhece essas "
                f"{subset} classes e vai forcar qualquer foto numa delas"
            )
            widget.set_class(True, "warn")

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#btn-predict", Button).disabled = busy
        self.query_one("#btn-sample", Button).disabled = busy
        self.query_one("#btn-file", Button).disabled = busy

    def _clear_result(self) -> None:
        for slot in range(TOP5_SLOTS):
            self.query_one(f"#top5-label-{slot}", Static).update("—")
            self.query_one(f"#top5-bar-{slot}", Static).update("")

    def _status(self, message: str, kind: str) -> None:
        widget = self.query_one("#predict-status", Static)
        widget.update(message)
        for name in ("ok", "warn", "err"):
            widget.set_class(name == kind, name)


# -- funcoes puras, usadas pela worker thread -----------------------------


def _load_dataset_sample(run_id: str) -> SampleOutcome:
    """Sorteia e prepara uma amostra sem carregar o checkpoint da rede."""
    import torch

    from cnnlab.core.data import CIFAR100_MEAN, CIFAR100_STD, load_data
    from cnnlab.core.predict import sample_from_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = read_config(run_id)
    data = load_data(spec.dataset, device)
    index = random.randrange(len(data.x_val))
    x, truth = sample_from_dataset(data, index)
    pixels = _pixels_from_tensor(x.detach().to("cpu"), CIFAR100_MEAN, CIFAR100_STD)
    return SampleOutcome(
        run_id=run_id,
        index=index,
        truth=truth,
        caption=f"amostra #{index} · {truth}",
        pixels=pixels,
    )


def _run_prediction(
    run_id: str,
    use_file: bool,
    raw_path: str,
    sample_index: int | None = None,
) -> PredictionOutcome:
    """Carrega o modelo e classifica. Roda inteira fora da UI thread."""
    import torch

    from cnnlab.core.data import CIFAR100_MEAN, CIFAR100_STD, load_data
    from cnnlab.core.predict import (
        load_model,
        predict_file,
        predict_tensor,
        sample_from_dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(run_id, device)

    if use_file:
        source = Path(raw_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"imagem nao encontrada: {source}")
        # A copia acontece antes da inferencia: e o arquivo de uploads/ que fica
        # registrado no predictions.jsonl.
        stored = _copy_to_uploads(source)
        pixels = _pixels_from_file(stored)
        prediction = predict_file(model, stored, class_names, device)
        caption = f"{stored.name} · 32x32 RGB"
        truth = None
        kind = "upload"
    else:
        if sample_index is None:
            raise ValueError("nenhuma amostra selecionada")
        spec = read_config(run_id)
        data = load_data(spec.dataset, device)
        x, truth = sample_from_dataset(data, sample_index)
        pixels = _pixels_from_tensor(x.detach().to("cpu"), CIFAR100_MEAN, CIFAR100_STD)
        prediction = predict_tensor(model, x, class_names)
        caption = f"amostra #{sample_index} · {truth}"
        kind = "dataset"

    return PredictionOutcome(
        run_id=run_id,
        source=kind,
        caption=caption,
        predicted=prediction.predicted,
        confidence=prediction.confidence,
        top5=list(prediction.top5),
        num_classes=len(class_names),
        truth=truth,
        pixels=pixels,
    )


def _copy_to_uploads(source: Path) -> Path:
    """Copia para `uploads/`, sem sobrescrever arquivo homonimo."""
    layout.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = layout.UPLOADS_DIR / source.name
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = layout.UPLOADS_DIR / f"{source.stem}-{stamp}{source.suffix}"
    shutil.copy2(source, destination)
    return destination


def _open_image_dialog() -> str:
    """Abre o seletor nativo de arquivos e devolve o caminho escolhido."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        return filedialog.askopenfilename(
            parent=root,
            title="Escolher imagem",
            filetypes=(
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Todos os arquivos", "*.*"),
            ),
        )
    finally:
        root.destroy()


def _pixels_from_file(path: Path) -> list[list[Pixel]]:
    """Miniatura do que a rede recebe: mesmo resize bilinear de `predict_file`."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        data = list(rgb.getdata())
    return [data[row * 32 : (row + 1) * 32] for row in range(32)]


def _pixels_from_tensor(x: Any, mean: tuple[float, float, float], std: tuple[float, float, float]) -> list[list[Pixel]]:
    """Desfaz a normalizacao para mostrar a amostra como ela e."""
    channels = []
    for index in range(3):
        channel = x[index].to("cpu").float() * std[index] + mean[index]
        channels.append(channel.clamp(0.0, 1.0).mul(255.0).round().tolist())

    height = len(channels[0])
    return [
        [
            (
                int(channels[0][row][col]),
                int(channels[1][row][col]),
                int(channels[2][row][col]),
            )
            for col in range(len(channels[0][row]))
        ]
        for row in range(height)
    ]


# -- formatacao -----------------------------------------------------------


def _bar(probability: float) -> str:
    filled = round(max(0.0, min(1.0, probability)) * BAR_WIDTH)
    return f"{'█' * filled}{'·' * (BAR_WIDTH - filled)}  {probability:.0%}"


def _fmt_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%d/%m %H:%M")
    except ValueError:
        return ts or "—"


def _fmt_image(image: Any) -> str:
    if not isinstance(image, str) or not image:
        return "amostra"
    return Path(image).name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Igual ao reader: linha invalida (a ultima, truncada) e ignorada."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows
