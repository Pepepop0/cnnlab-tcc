"""Tela Config: preferencias da maquina, nao da run.

Nesta rodada os valores sao fixos e o botao nao grava nada. No wiring vira
`settings.json` na raiz, com write atomico.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, Switch

from cnnlab.tui.widgets.forms import labeled, section

FAKE_DEVICES = [("cuda:0 · RTX 5070 (12 GB)", "cuda:0"), ("cpu", "cpu")]


class SettingsView(Horizontal):
    """Sidebar com os campos, area principal com o resumo do ambiente."""

    DEFAULT_CLASSES = "view"

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="sidebar"):
            yield section("DATASET", first=True)
            yield from labeled("caminho", Input(value="./data", id="cfg-root", compact=True))

            yield section("EXECUCAO")
            yield from labeled(
                "device",
                Select(
                    FAKE_DEVICES,
                    value="cuda:0",
                    allow_blank=False,
                    id="cfg-device",
                    compact=True,
                ),
            )
            yield from labeled("amp", Switch(value=True, id="cfg-amp"))
            yield from labeled("channels_last", Switch(value=True, id="cfg-channels-last"))

            with Horizontal(classes="buttons"):
                yield Button("Salvar", id="btn-save", classes="-primary")
            yield Static("Gravado em settings.json na raiz.", classes="hint")

        with VerticalScroll(classes="main"):
            with Vertical(id="env", classes="card"):
                yield Static("Ambiente", classes="panel-title")
                yield Static(
                    "GPU        RTX 5070 · 12 GB · sm_120\n"
                    "torch      2.13.0+cu130\n"
                    "python     3.14\n"
                    "runs/      5 pastas · 41 MB",
                    id="env-info",
                )
            with Vertical(id="notes", classes="card"):
                yield Static("Notas", classes="panel-title")
                yield Static(
                    "O dataset inteiro vive na VRAM (decisao D5): nao ha DataLoader.\nSem temas nesta POC.",
                    classes="hint",
                )

    def on_mount(self) -> None:
        self.query_one("#env").border_title = "Maquina"
        self.query_one("#notes").border_title = "Lembretes"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """TODO(wiring): gravar `settings.json` com write atomico."""
        event.stop()
