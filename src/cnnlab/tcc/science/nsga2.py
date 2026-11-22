"""NSGA-II ranking on (E, L, V, S), all minimized (Deb et al. 2002; TCC-01 section 4).

Pure functions over candidate ids and optional objective vectors. Invalid
candidates (``None`` objectives) are dominated by every valid one
(constraint-domination: feasible solutions always win) and form the last
front, where they tie. Nothing here invents a value for an invalid
candidate: it is only ranked behind the valid ones.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cnnlab.tcc.science.sampling import GeneRng

Objectives = tuple[float, ...]


def dominates(a: Objectives, b: Objectives) -> bool:
    """``a`` is no worse in every objective and strictly better in one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_fronts(points: Mapping[str, Objectives | None]) -> list[list[str]]:
    """Fast non-dominated sort; invalid ids (``None``) go to one final front.

    Ids inside a front keep the insertion order of ``points`` so results are
    deterministic.
    """
    valid = [k for k, v in points.items() if v is not None]
    invalid = [k for k, v in points.items() if v is None]
    dominated_by: dict[str, list[str]] = {k: [] for k in valid}
    count: dict[str, int] = {k: 0 for k in valid}
    for i, p in enumerate(valid):
        for q in valid[i + 1 :]:
            fp, fq = points[p], points[q]
            assert fp is not None and fq is not None
            if dominates(fp, fq):
                dominated_by[p].append(q)
                count[q] += 1
            elif dominates(fq, fp):
                dominated_by[q].append(p)
                count[p] += 1
    fronts: list[list[str]] = []
    current = [k for k in valid if count[k] == 0]
    while current:
        fronts.append(current)
        nxt: list[str] = []
        for p in current:
            for q in dominated_by[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(q)
        order = {k: i for i, k in enumerate(valid)}
        current = sorted(nxt, key=order.__getitem__)
    if invalid:
        fronts.append(invalid)
    return fronts


def crowding_distance(front: Sequence[str], points: Mapping[str, Objectives | None]) -> dict[str, float]:
    """Crowding distance inside one front; boundary points get infinity, invalid ids 0."""
    distance = {k: 0.0 for k in front}
    valid = [k for k in front if points[k] is not None]
    if len(valid) <= 2:
        for k in valid:
            distance[k] = math.inf
        return distance
    n_obj = len(points[valid[0]])  # type: ignore[arg-type]
    for m in range(n_obj):
        ordered = sorted(valid, key=lambda k: (points[k][m], k))  # type: ignore[index]
        low, high = points[ordered[0]][m], points[ordered[-1]][m]  # type: ignore[index]
        distance[ordered[0]] = distance[ordered[-1]] = math.inf
        if high == low:
            continue
        for prev, cur, nxt in zip(ordered, ordered[1:], ordered[2:]):
            if distance[cur] != math.inf:
                distance[cur] += (points[nxt][m] - points[prev][m]) / (high - low)  # type: ignore[index]
    return distance


def rank_and_crowding(points: Mapping[str, Objectives | None]) -> dict[str, tuple[int, float]]:
    """``(front rank, crowding distance)`` per id; lower rank then larger crowding is better."""
    out: dict[str, tuple[int, float]] = {}
    for rank, front in enumerate(non_dominated_fronts(points)):
        for k, d in crowding_distance(front, points).items():
            out[k] = (rank, d)
    return out


def _better(a: str, b: str, ranks: Mapping[str, tuple[int, float]]) -> bool:
    (ra, da), (rb, db) = ranks[a], ranks[b]
    return ra < rb or (ra == rb and da > db)


def tournament(ids: Sequence[str], ranks: Mapping[str, tuple[int, float]], rng: GeneRng, n: int) -> list[str]:
    """``n`` parents by binary tournament on (rank, crowding); ties go to the first drawn."""
    parents: list[str] = []
    for _ in range(n):
        a = ids[rng.index(len(ids))]
        b = ids[rng.index(len(ids))]
        parents.append(b if _better(b, a, ranks) else a)
    return parents


def environmental_selection(points: Mapping[str, Objectives | None], size: int) -> list[str]:
    """Keep ``size`` ids: whole fronts first, the last partial front by crowding distance."""
    survivors: list[str] = []
    for front in non_dominated_fronts(points):
        if len(survivors) + len(front) <= size:
            survivors += front
            continue
        crowd = crowding_distance(front, points)
        order = {k: i for i, k in enumerate(front)}
        rest = sorted(front, key=lambda k: (-crowd[k], order[k]))
        survivors += rest[: size - len(survivors)]
        break
    return survivors


def pareto_archive(points: Mapping[str, Objectives | None]) -> list[str]:
    """Non-dominated valid ids among everything evaluated (the external archive)."""
    fronts = non_dominated_fronts({k: v for k, v in points.items() if v is not None})
    return fronts[0] if fronts else []
