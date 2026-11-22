"""Statistics of the pilot report (tools/piloto), checked against hand-computed values."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "piloto"))
from estatistica import pearson, percentile, ranks, spearman  # noqa: E402


def test_ranks_average_ties() -> None:
    assert ranks([10, 20, 20, 5]) == [2.0, 3.5, 3.5, 1.0]


def test_spearman_known_values() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # d = (0, 0, 1, -1) -> rho = 1 - 6*2 / (4*15) = 0.8
    assert spearman([0.20, 0.25, 0.18, 0.30], [0.40, 0.48, 0.41, 0.55]) == pytest.approx(0.8)


def test_pearson_undefined_cases() -> None:
    assert pearson([1.0], [2.0]) is None
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        pearson([1, 2], [1])


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile([0, 10], 90) == 9


def test_lock_removes_an_orphan_and_respects_a_live_holder(tmp_path: Path) -> None:
    import subprocess

    import trava

    lock = tmp_path / "gpu.lock"
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    lock.write_text(f"owner=x\npid={int(dead.stdout)}\njob=morreu\n", encoding="utf-8")
    trava.take("teste", "pega", lock=lock)  # orphan removed, lock taken
    assert trava.lock_pid(lock.read_text(encoding="utf-8")) == os.getpid()
    trava.release(lock=lock)
    assert not lock.exists()

    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(f"owner=y\npid={live.pid}\njob=vivo\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="GPU travada"):
            trava.take("teste", "pega", lock=lock)
        trava.release(lock=lock)  # not ours: untouched
        assert lock.exists()
    finally:
        live.kill()

    lock.write_text("owner=sem pid\njob=outra ferramenta\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        trava.take("teste", "pega", lock=lock)
