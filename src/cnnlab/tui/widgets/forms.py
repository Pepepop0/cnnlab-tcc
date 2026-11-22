"""Helpers de composicao compartilhados pelas telas."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, Static


def section(title: str, *, first: bool = False) -> Static:
    """Titulo de bloco dentro da sidebar."""
    classes = "section section-first" if first else "section"
    return Static(title, classes=classes)


def labeled(label: str, widget: Widget) -> ComposeResult:
    """Uma linha de formulario: rotulo a esquerda, controle a direita."""
    with Horizontal(classes="field"):
        yield Label(label, classes="field-label")
        yield widget
