from __future__ import annotations

import pytest

from cnnlab.tcc.contracts import EvaluationResult, Fitness, TCCJobSpec


def test_invalid_fitness_rejected() -> None:
    with pytest.raises(ValueError):
        Fitness(e=0.2, l_ms=0.0, v_mb=1.0, s_mb=1.0)


def test_invalid_attempt_has_no_fitness() -> None:
    with pytest.raises(ValueError):
        EvaluationResult("r", "c", "a", "invalid", Fitness(0.2, 1, 1, 1), 0)


def test_confirmatory_budget_is_fixed() -> None:
    with pytest.raises(ValueError):
        TCCJobSpec("r", "p", "plan", "d", "env", "space", "joint", "confirmatory", 0, 1, 2, 3, 100)
