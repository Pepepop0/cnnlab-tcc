"""Exact hypervolume of a minimization front, for any number of objectives (TCC-01 sections 4 and 5).

Own implementation, no dependency: the project does not install pymoo. The volume is the
Lebesgue measure of the region dominated by the points and bounded by the reference point,
computed exactly (up to floating point) by slicing the last objective: between two
consecutive values of the last coordinate the dominated cross-section is the hypervolume
of the points seen so far in one dimension fewer. The base case is the area under a
staircase. Pure Python; measured on random 4-objective fronts: 100 points in 0.03 s,
300 points in 0.3 s.

Conventions fixed here:

* Every objective is minimized. The vectors are ``(E, L, V, S)`` in this project: top-1
  error as a fraction, latency in ms, peak memory and engine size in decimal MB.
* A point contributes only if it is strictly below the reference point in **every**
  coordinate. One at or beyond it in any coordinate adds no volume. Such points are not
  dropped silently: :func:`hypervolume_of` lists them (the protocol asks for them to be
  counted and listed).
* Candidates without a valid measurement (``None``) are ignored, never turned into a number.
* The **official** value is normalized (proposal 5.1.3): every objective is divided by its
  reference coordinate, ``F~_j = F_j / r_j``, which puts the reference point at (1, ..., 1) and the
  hypervolume in [0, 1]. Dividing the volume by a constant leaves differences, ranks and the
  paired permutation test as they are; it changes the number that goes to the text. It equals the
  raw volume divided by ``prod(r_j)``, so it does not depend on the units of E, L, V and S.
  :func:`hypervolume` returns the **raw** volume (units E x ms x MB x MB); wherever both are
  stored the raw one is labeled as such. Methods are comparable because every one is scored
  with the same reference point (TCC-01 section 5), fixed once after the pilot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Point = tuple[float, ...]

# Position of each objective in a vector, and the protocol's reference-point factors (TCC-01 section 5).
OBJECTIVE_NAMES = ("E", "L", "V", "S")
REFERENCE_E = 1.0
REFERENCE_FACTOR = 1.25
SENSITIVITY_FACTOR = 1.5


@dataclass(frozen=True, slots=True)
class HypervolumeResult:
    """The normalized volume (the official value) and what it was computed from.

    ``value`` is ``raw / prod(reference)``, in [0, 1]; ``raw`` is the volume in the units of the
    objectives. ``n_front`` counts the non-dominated points that lie inside the reference box and
    so shape the volume. ``outside`` lists the ids of valid candidates at or beyond the reference
    point in some coordinate (they add nothing); it is empty when ids were not given.
    """

    value: float
    reference: tuple[float, ...]
    n_points: int
    n_front: int
    outside: tuple[str, ...]
    raw: float = 0.0

    @property
    def scale(self) -> float:
        """The product of the reference coordinates: raw volume / scale = the official value."""
        return math.prod(self.reference)


def _vector(values: Sequence[float], size: int, what: str) -> Point:
    if len(values) != size:
        raise ValueError(f"{what} has {len(values)} coordinates, expected {size}")
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} has a non-finite coordinate: {out}")
    return out


def _non_dominated(points: Sequence[Point]) -> list[Point]:
    """The points no other point is no worse than in every objective (duplicates kept once)."""
    kept: list[Point] = []
    for p in sorted(set(points)):  # lexicographic: only an earlier point can dominate a later one
        if not any(all(q[i] <= p[i] for i in range(len(p))) for q in kept):
            kept.append(p)
    return kept


def _area(points: Sequence[Point], ref: Point) -> float:
    """Dominated area of 2-D points below ``ref``: the area under a staircase."""
    ordered = sorted(points)
    area, floor = 0.0, ref[1]
    for i, (x, y) in enumerate(ordered):
        floor = min(floor, y)
        right = ordered[i + 1][0] if i + 1 < len(ordered) else ref[0]
        area += (right - x) * (ref[1] - floor)
    return area


def _volume(points: Sequence[Point], ref: Point) -> float:
    """``points`` are inside the box and non-dominated; slice the last objective."""
    if not points:
        return 0.0
    dims = len(ref)
    if dims == 1:
        return ref[0] - min(p[0] for p in points)
    if dims == 2:
        return _area(points, ref)
    ordered = sorted(points, key=lambda p: p[-1])
    total = 0.0
    section: list[Point] = []  # non-dominated projections (last coordinate dropped) of the points seen so far
    for i, p in enumerate(ordered):
        q = p[:-1]
        if not any(all(s[j] <= q[j] for j in range(len(q))) for s in section):  # not covered by what is in
            section = [s for s in section if not all(q[j] <= s[j] for j in range(len(q)))]
            section.append(q)
        thickness = (ordered[i + 1][-1] if i + 1 < len(ordered) else ref[-1]) - p[-1]
        if thickness > 0.0:  # a zero-thickness slab is counted with the next point of the same last value
            total += _volume(section, ref[:-1]) * thickness
    return total


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """Volume dominated by ``points`` inside the box that ends at ``reference`` (all minimized)."""
    if not reference:
        raise ValueError("the reference point needs at least one coordinate")
    ref = _vector(reference, len(reference), "reference point")
    inside = [p for p in (_vector(point, len(ref), "point") for point in points) if all(x < r for x, r in zip(p, ref))]
    return _volume(_non_dominated(inside), ref)


def _scale(reference: Point) -> float:
    if not all(r > 0 for r in reference):
        raise ValueError("normalizing divides by the reference point: every coordinate must be positive")
    return math.prod(reference)


def normalized_hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """The official value: each objective divided by its reference coordinate, a number in [0, 1]."""
    ref = _vector(reference, len(reference), "reference point")
    return hypervolume(points, ref) / _scale(ref)


def hypervolume_of(objectives: Mapping[str, Sequence[float] | None], reference: Sequence[float]) -> HypervolumeResult:
    """Normalized hypervolume (and the raw volume) of keyed vectors, listing candidates beyond the point.

    ``None`` values (no valid measurement) are ignored. Give every valid evaluated candidate, not
    only the front: a point beyond the reference point can never dominate one inside the box, so
    the volume is the same, and the list of candidates beyond the point is then complete.
    """
    ref = _vector(reference, len(reference), "reference point")
    valid = {k: _vector(v, len(ref), f"objectives of {k!r}") for k, v in objectives.items() if v is not None}
    outside = tuple(sorted(k for k, v in valid.items() if not all(x < r for x, r in zip(v, ref))))
    inside = [v for k, v in valid.items() if k not in set(outside)]
    front = _non_dominated(inside)
    raw = _volume(front, ref)
    return HypervolumeResult(
        value=raw / _scale(ref),
        reference=ref,
        n_points=len(valid),
        n_front=len(front),
        outside=outside,
        raw=raw,
    )


def reference_from_pilot(
    pilot_points: Sequence[Sequence[float]], *, factor: float = REFERENCE_FACTOR, error: float = REFERENCE_E
) -> tuple[float, float, float, float]:
    """The reference point of TCC-01 section 5: ``r_E = 1`` and ``r_j = 1.25 * max_j`` for L, V, S.

    ``pilot_points`` are the ``(E, L, V, S)`` vectors of the pilot's viable genomes.
    """
    if not pilot_points:
        raise ValueError("the reference point needs at least one viable pilot point")
    rows = [_vector(p, len(OBJECTIVE_NAMES), "pilot point") for p in pilot_points]
    return (float(error), *(factor * max(row[j] for row in rows) for j in (1, 2, 3)))


def sensitivity_reference(
    reference: Sequence[float], *, factor: float = SENSITIVITY_FACTOR
) -> tuple[float, float, float, float]:
    """``r'_j = 1.5 * r_j`` for L, V, S (``r_E`` stays), the sensitivity analysis of TCC-01 section 5."""
    r = _vector(reference, len(OBJECTIVE_NAMES), "reference point")
    return (r[0], *(factor * r[j] for j in (1, 2, 3)))
