"""NSGA-II ranking: cross-checked against DEAP 1.4 where the algorithms agree,
and against explicit invariants where they intentionally diverge (see the
crowding-distance normalization note below).
"""

from __future__ import annotations

import math
import random

import pytest
from deap import base as deap_base
from deap import tools as deap_tools

from cnnlab.tcc.science.nsga2 import (
    Objectives,
    crowding_distance,
    dominates,
    environmental_selection,
    non_dominated_fronts,
    pareto_archive,
    rank_and_crowding,
    tournament,
)
from cnnlab.tcc.science.sampling import GeneRng

N_OBJ = 4


class _MinFitness(deap_base.Fitness):
    weights = (-1.0, -1.0, -1.0, -1.0)


class _Ind(list):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__(values)
        self.fitness = _MinFitness()
        self.fitness.values = values


def _random_points(seed: int, n: int) -> dict[str, Objectives]:
    """``n`` random 4-objective points, seeded, with intentional ties/duplicates."""
    rng = random.Random(seed)
    points: dict[str, Objectives] = {}
    pool: list[tuple[float, ...]] = []
    for i in range(n):
        if pool and rng.random() < 0.3:
            # Duplicate an earlier point (exact tie) to exercise tie handling.
            values = pool[rng.randrange(len(pool))]
        else:
            # Coarse rounding raises the odds of a tie on a single objective too.
            values = tuple(round(rng.uniform(0.0, 5.0), 1) for _ in range(N_OBJ))
        pool.append(values)
        points[f"c{i}"] = values
    return points


def _deap_fronts(points: dict[str, Objectives]) -> list[set[str]]:
    ids = list(points)
    inds = [_Ind(points[k]) for k in ids]
    id_of = {id(ind): k for ind, k in zip(inds, ids)}
    fronts = deap_tools.sortNondominated(inds, k=len(inds), first_front_only=False)
    return [set(id_of[id(ind)] for ind in front) for front in fronts]


def _deap_crowding(points: dict[str, Objectives]) -> dict[str, float]:
    ids = list(points)
    inds = [_Ind(points[k]) for k in ids]
    deap_tools.emo.assignCrowdingDist(inds)
    return {k: ind.fitness.crowding_dist for k, ind in zip(ids, inds)}


# --- dominates ---------------------------------------------------------------


def test_dominates_matches_deap_pairwise_definition() -> None:
    rng = random.Random(0)
    for _ in range(200):
        a = tuple(round(rng.uniform(0, 5), 1) for _ in range(N_OBJ))
        b = tuple(round(rng.uniform(0, 5), 1) for _ in range(N_OBJ))
        ia, ib = _Ind(a), _Ind(b)
        assert dominates(a, b) == ia.fitness.dominates(ib.fitness)


# --- non_dominated_fronts ------------------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_front_partition_matches_deap(seed: int) -> None:
    n = 4 + seed % 12
    points = _random_points(seed, n)
    ours = [set(front) for front in non_dominated_fronts(points)]
    assert ours == _deap_fronts(points)


def test_invalid_ids_form_one_last_front_and_rank_behind_every_valid() -> None:
    points: dict[str, Objectives | None] = {
        "v1": (1.0, 1.0, 1.0, 1.0),
        "i1": None,
        "v2": (2.0, 2.0, 2.0, 2.0),
        "i2": None,
        "v3": (0.5, 5.0, 1.0, 1.0),
    }
    fronts = non_dominated_fronts(points)
    assert set(fronts[-1]) == {"i1", "i2"}
    assert all("i1" not in f and "i2" not in f for f in fronts[:-1])
    ranks = rank_and_crowding(points)
    invalid_rank = ranks["i1"][0]
    assert ranks["i2"][0] == invalid_rank
    assert all(ranks[k][0] < invalid_rank for k in ("v1", "v2", "v3"))


def test_all_invalid_is_the_only_front() -> None:
    points: dict[str, Objectives | None] = {"a": None, "b": None}
    assert non_dominated_fronts(points) == [["a", "b"]]


def test_front_order_is_insertion_order_of_points() -> None:
    # Three mutually non-dominating points share the first front; order must
    # follow insertion order of the input mapping, not sorted id order.
    points: dict[str, Objectives] = {
        "z": (0.0, 1.0, 1.0, 1.0),
        "a": (1.0, 0.0, 1.0, 1.0),
        "m": (1.0, 1.0, 0.0, 1.0),
    }
    assert non_dominated_fronts(points)[0] == ["z", "a", "m"]


def test_empty_points() -> None:
    assert non_dominated_fronts({}) == []


# --- crowding_distance ---------------------------------------------------------


def test_boundary_points_are_infinite_and_front_of_two_or_fewer_is_all_infinite() -> None:
    points: dict[str, Objectives] = {"a": (0.0, 0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0, 1.0)}
    dist = crowding_distance(["a", "b"], points)
    assert dist == {"a": math.inf, "b": math.inf}


def test_crowding_distance_matches_deap_up_to_the_nobj_normalization() -> None:
    """DEAP's ``assignCrowdingDist`` divides each objective's term by ``nobj``
    (see ``deap/tools/emo.py``), on top of the ``(high - low)`` range that the
    Deb et al. (2002) formula itself calls for. ``nsga2.crowding_distance``
    follows the textbook formula without that extra factor (matching e.g.
    pymoo), so for points with no duplicate objective values the two are
    related by a constant factor of ``N_OBJ`` rather than being equal outright.
    Ties are handled separately below because DEAP's neighbor bookkeeping
    depends on which duplicate lands where in its internal sort, which is not
    part of the published algorithm.
    """
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(4, 10)
        # Full float precision keeps every objective value unique.
        points = {f"c{i}": tuple(rng.uniform(0.0, 5.0) for _ in range(N_OBJ)) for i in range(n)}
        ours = crowding_distance(list(points), points)
        theirs = _deap_crowding(points)
        for k in points:
            if math.isinf(ours[k]):
                assert math.isinf(theirs[k])
            else:
                assert ours[k] == pytest.approx(N_OBJ * theirs[k])


def test_crowding_distance_with_ties_stays_finite_and_nonnegative() -> None:
    """With duplicate objective values, DEAP's and our distances need not match
    exactly (see the test above); we instead check the invariants that
    ``rank_and_crowding``/``environmental_selection`` actually rely on.
    """
    points: dict[str, Objectives] = {
        "a": (1.0, 1.0, 1.0, 1.0),
        "b": (1.0, 1.0, 1.0, 1.0),  # exact duplicate of a
        "c": (2.0, 2.0, 2.0, 2.0),
        "d": (3.0, 3.0, 3.0, 3.0),
    }
    dist = crowding_distance(list(points), points)
    assert dist["a"] == math.inf and dist["d"] == math.inf  # boundary on every objective
    assert dist["b"] >= 0.0 and math.isfinite(dist["b"])
    assert dist["c"] >= 0.0 and math.isfinite(dist["c"])


def test_invalid_points_get_zero_crowding_distance() -> None:
    points: dict[str, Objectives | None] = {
        "a": (0.0, 0.0, 0.0, 0.0),
        "b": (1.0, 1.0, 1.0, 1.0),
        "c": (2.0, 2.0, 2.0, 2.0),
        "i": None,
    }
    dist = crowding_distance(["a", "b", "c", "i"], points)
    assert dist["i"] == 0.0


# --- rank_and_crowding / tournament --------------------------------------------


def test_tournament_is_deterministic_for_a_given_rng_seed() -> None:
    ids = [f"c{i}" for i in range(10)]
    points = _random_points(11, 10)
    ranks = rank_and_crowding(points)
    a = tournament(ids, ranks, GeneRng(42), 20)
    b = tournament(ids, ranks, GeneRng(42), 20)
    assert a == b


class _AlternatingRng:
    """Fake RNG whose ``.index`` alternates 0, 1, 0, 1, ... so a tournament
    always draws one of each candidate (real ``GeneRng`` could draw the same
    id twice, which is a legitimate tie, not what this test is about)."""

    def __init__(self) -> None:
        self._i = 0

    def index(self, n: int) -> int:
        value = self._i % 2
        self._i += 1
        return value


def test_tournament_never_prefers_a_strictly_worse_candidate() -> None:
    points: dict[str, Objectives] = {
        "best": (0.0, 0.0, 0.0, 0.0),
        "worst": (5.0, 5.0, 5.0, 5.0),
    }
    ranks = rank_and_crowding(points)
    winners = tournament(["best", "worst"], ranks, _AlternatingRng(), 50)
    assert all(w == "best" for w in winners)


# --- environmental_selection ---------------------------------------------------


def test_environmental_selection_keeps_whole_fronts_then_crowding() -> None:
    # Two 4-objective Pareto layers (dims 2/3 are 10x copies of dims 0/1, so
    # every objective varies and no accidental ties confuse crowding).
    points: dict[str, Objectives] = {
        "f0a": (0.0, 6.0, 0.0, 60.0),
        "f0b": (2.0, 3.0, 20.0, 30.0),
        "f0c": (5.0, 0.0, 50.0, 0.0),
        "f1a": (3.0, 7.0, 30.0, 70.0),  # dominated by f0a
        "f1b": (6.0, 3.0, 60.0, 30.0),  # dominated by f0c
        "f1c": (4.0, 4.0, 40.0, 40.0),  # dominated by f0b
    }
    fronts = non_dominated_fronts(points)
    assert len(fronts[0]) == 3 and len(fronts[1]) == 3
    survivors = environmental_selection(points, 4)
    assert set(fronts[0]) <= set(survivors)
    assert len(survivors) == 4
    # Exactly one member of the partial front survives, and it is not the one
    # with the (uniquely) lowest crowding distance.
    crowd = crowding_distance(fronts[1], points)
    kept_from_f1 = [k for k in survivors if k in fronts[1]]
    assert kept_from_f1 == ["f1a"]
    dropped = set(fronts[1]) - set(kept_from_f1)
    assert all(crowd[k] >= crowd[d] for k in kept_from_f1 for d in dropped)


def test_environmental_selection_excludes_invalid_when_room_is_short() -> None:
    points: dict[str, Objectives | None] = {
        "v1": (0.0, 0.0, 0.0, 0.0),
        "v2": (1.0, 1.0, 1.0, 1.0),
        "i1": None,
    }
    assert set(environmental_selection(points, 2)) == {"v1", "v2"}


def test_environmental_selection_size_larger_than_population_keeps_everyone() -> None:
    points: dict[str, Objectives | None] = {"v1": (0.0, 0.0, 0.0, 0.0), "i1": None}
    assert set(environmental_selection(points, 10)) == {"v1", "i1"}


# --- pareto_archive --------------------------------------------------------------


def test_pareto_archive_excludes_invalid_and_dominated() -> None:
    points: dict[str, Objectives | None] = {
        "dominated": (2.0, 2.0, 2.0, 2.0),
        "nondominated": (1.0, 1.0, 1.0, 1.0),
        "invalid": None,
    }
    assert pareto_archive(points) == ["nondominated"]


def test_pareto_archive_of_empty_is_empty() -> None:
    assert pareto_archive({}) == []


@pytest.mark.parametrize("seed", range(10))
def test_pareto_archive_matches_deap_first_front(seed: int) -> None:
    points = _random_points(100 + seed, 8)
    assert set(pareto_archive(points)) == _deap_fronts(points)[0]
