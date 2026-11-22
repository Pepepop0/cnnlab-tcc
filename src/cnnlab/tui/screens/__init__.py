"""As quatro telas.

Cada uma e um container horizontal (sidebar + area principal) trocado por um
`ContentSwitcher`, nao uma `Screen` do Textual: a nav e a barra de status
precisam continuar visiveis em todas.
"""

from __future__ import annotations

from cnnlab.tui.screens.history import HistoryView
from cnnlab.tui.screens.predict import PredictView
from cnnlab.tui.screens.settings import SettingsView
from cnnlab.tui.screens.train import TrainView

__all__ = ["HistoryView", "PredictView", "SettingsView", "TrainView"]
