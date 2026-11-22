"""Casca da interface e dona do estado compartilhado.

O app possui a fila, os dois `threading.Event` de controle e a worker thread
que roda o `Runner`. As telas nao falam com `worker/` nem com `core/`: pedem ao
app por `enqueue`, `cancel_current` e `ensure_runner`.

Regras de threading (ver `docs/01-ARCHITECTURE.md`):

- `torch` e `worker.runner` sao importados **dentro** da worker thread. No topo
  do modulo eles custariam segundos de espera antes do primeiro frame.
- a worker thread nunca toca widget: so chama `post_message`, que o Textual
  garante ser seguro entre threads.
- quem atualiza widget e o handler `on_core_event`, que roda na UI thread.
"""

from __future__ import annotations

import threading

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import ContentSwitcher, Static, Tab, Tabs

from cnnlab.config import JobSpec
from cnnlab.events import Event, RunCancelled, RunFailed, RunFinished, RunStarted
from cnnlab.tui.screens import HistoryView, PredictView, SettingsView, TrainView
from cnnlab.tui.screens.train import CoreEvent
from cnnlab.worker.queue import JobQueue

# Ordem das abas: id do view, rotulo. O id da Tab e "tab-{id do view}".
TABS: tuple[tuple[str, str], ...] = (
    ("train", "Treinar"),
    ("history", "Historico"),
    ("predict", "Usar rede"),
    ("settings", "Config"),
)


class DeviceReady(Message):
    """Nome do device, descoberto na worker thread junto com o import do torch."""

    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__()


class StatusBar(Horizontal):
    """Barra inferior. `set_state` so pode ser chamada da UI thread."""

    def compose(self) -> ComposeResult:
        yield Static("status: ocioso | fila: 0 | GPU: —", id="status-left")
        yield Static("1-4 abas · q sair", id="status-right")

    def set_state(self, status: str, queued: int, gpu: str) -> None:
        self.query_one("#status-left", Static).update(f"status: {status} | fila: {queued} | GPU: {gpu}")


class TrainingApp(App[None]):
    """Aplicacao. Uma tela so: nav e barra de status ficam sempre visiveis."""

    CSS_PATH = "app.tcss"
    TITLE = "CNN Lab TCC"

    # Terminal estreito empilha sidebar sobre a area principal (ver app.tcss).
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]

    BINDINGS = [
        Binding("q", "quit", "Sair"),
        Binding("1", "show_tab('train')", "Treinar"),
        Binding("2", "show_tab('history')", "Historico"),
        Binding("3", "show_tab('predict')", "Usar rede"),
        Binding("4", "show_tab('settings')", "Config"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.queue = JobQueue()
        # `stop` encerra o loop depois do job atual; `cancel_current` aborta so
        # o job em execucao. Sao coisas diferentes de proposito.
        self._stop = threading.Event()
        self._cancel_current = threading.Event()
        self._runner_started = False
        self._device_label = "detectando…"
        self._status_text = "ocioso"

    def compose(self) -> ComposeResult:
        yield Static("CNN LAB TCC   ", id="titlebar")
        yield Tabs(
            *(Tab(label, id=f"tab-{view_id}") for view_id, label in TABS),
            id="nav",
        )
        with ContentSwitcher(initial="train", id="body"):
            yield TrainView(id="train")
            yield HistoryView(id="history")
            yield PredictView(id="predict")
            yield SettingsView(id="settings")
        yield StatusBar(id="statusbar")

    def on_mount(self) -> None:
        """Recupera a fila do disco e sobe o worker. Roda na UI thread."""
        try:
            self.queue.load()
        except (OSError, ValueError) as exc:
            self.query_one("#train", TrainView).log_line(f"[b red]fila ilegivel[/] {type(exc).__name__}: {exc}")
        self.ensure_runner()
        self.refresh_status()

    def on_unmount(self) -> None:
        """Fecha o worker ao sair.

        `cancel_current` junto com `stop`: sem ele, sair durante um retreino de
        25 epocas seguraria o fechamento por minutos. Com ele, o corte vem em
        ate uma epoca.
        """
        self._cancel_current.set()
        self._stop.set()

    # -- servicos para as telas -------------------------------------------

    def ensure_runner(self) -> None:
        """Sobe a worker thread se ela ainda nao estiver de pe. Idempotente."""
        if self._runner_started:
            return
        self._runner_started = True
        self.start_runner()

    def enqueue(self, specs: list[JobSpec]) -> None:
        """Acrescenta jobs a fila persistente e atualiza a barra de status."""
        self.queue.add_many(specs)
        self.refresh_status("na fila" if self._status_text == "ocioso" else None)

    def cancel_current(self) -> None:
        """Aborta o job em execucao. A fila continua."""
        self._cancel_current.set()
        self.refresh_status("cancelando")

    def refresh_status(self, status: str | None = None) -> None:
        if status is not None:
            self._status_text = status
        self.query_one("#statusbar", StatusBar).set_state(
            self._status_text, self.queue.pending_count(), self._device_label
        )

    # -- worker thread -----------------------------------------------------

    @work(thread=True, exclusive=True, group="runner")
    def start_runner(self) -> None:
        """Consome a fila ate `stop`. Nao toca em widget: so `post_message`."""
        import torch

        from cnnlab.worker.runner import Runner

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        label = torch.cuda.get_device_name(0) if use_cuda else "CPU (sem CUDA)"
        self.post_message(DeviceReady(label))

        runner = Runner(self.queue, self._post_core_event, device)
        runner.run_forever(self._stop, self._cancel_current)

    def _post_core_event(self, evt: Event) -> None:
        """Callback entregue ao `Runner`. Chamado de dentro da worker thread."""
        self.post_message(CoreEvent(evt))

    # -- mensagens ---------------------------------------------------------

    def on_device_ready(self, message: DeviceReady) -> None:
        self._device_label = message.label
        self.refresh_status()

    def on_core_event(self, message: CoreEvent) -> None:
        """Unico ponto em que evento do core vira mudanca de widget."""
        evt = message.evt
        self.query_one("#train", TrainView).apply_core_event(evt)

        if isinstance(evt, RunStarted):
            self.refresh_status("rodando")
        elif isinstance(evt, (RunFinished, RunFailed, RunCancelled)):
            self._cancel_current.clear()
            self.refresh_status("ocioso" if self.queue.pending_count() == 0 else "na fila")
        else:
            self.refresh_status()

    # -- navegacao ---------------------------------------------------------

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Troca o painel central. `Tabs` guarda o id com prefixo `tab-`."""
        if event.tab is None:
            return
        self.query_one("#body", ContentSwitcher).current = event.tab.id.removeprefix("tab-")

    def action_show_tab(self, view_id: str) -> None:
        self.query_one("#nav", Tabs).active = f"tab-{view_id}"


def main() -> None:
    """Sobe a TUI."""
    TrainingApp().run()


if __name__ == "__main__":
    main()
