"""Miniatura da imagem classificada, em meio-bloco.

Cada celula do terminal vira dois pixels: `▀` com a cor de cima no texto e a de
baixo no fundo. Uma imagem 32x32 cabe em 32 colunas por 16 linhas, que e o
tamanho real do que a rede ve — util justamente porque revela quando a imagem
enviada virou uma papa de 32x32.

As cores do DEFAULT_CSS espelham a paleta de `app.tcss` de proposito: o widget
carrega o proprio estilo e nao depende de regra externa.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

Pixel = tuple[int, int, int]
HALF_BLOCK = "▀"


class Thumbnail(Static):
    """Static que sabe desenhar uma matriz de pixels."""

    DEFAULT_CSS = """
    Thumbnail {
        /* 16 linhas de meio-bloco + legenda + borda + padding. */
        height: 21;
        width: 1fr;
        padding: 1 0;
        content-align: center middle;
        text-align: center;
        color: #7d8b9a;
        background: #131a22;
        border: round #253243;
        border-title-color: #4fd6be;
    }
    """

    def show_pixels(self, rows: list[list[Pixel]], caption: str = "") -> None:
        """Desenha a imagem. `rows` e uma matriz RGB de altura par."""
        self.update(_render(rows, caption))

    def show_message(self, message: str) -> None:
        """Estado vazio: sem imagem ainda, ou erro ao abrir."""
        self.update(message)


def _render(rows: list[list[Pixel]], caption: str) -> Text:
    text = Text(no_wrap=True, overflow="crop")
    for top_row, bottom_row in zip(rows[::2], rows[1::2], strict=False):
        for top, bottom in zip(top_row, bottom_row, strict=False):
            text.append(
                HALF_BLOCK,
                Style(color=_rgb(top), bgcolor=_rgb(bottom)),
            )
        text.append("\n")
    if caption:
        text.append(caption, Style(color="#7d8b9a"))
    return text


def _rgb(pixel: Pixel) -> str:
    red, green, blue = (max(0, min(255, int(channel))) for channel in pixel)
    return f"rgb({red},{green},{blue})"
