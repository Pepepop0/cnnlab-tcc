"""Exact hypervolume: cases worked out by hand, plus an independent oracle (inclusion-exclusion)."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from cnnlab.tcc.science.hypervolume import (
    HypervolumeResult,
    hypervolume,
    hypervolume_of,
    normalized_hypervolume,
    reference_from_pilot,
    sensitivity_reference,
)


def oracle(points: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    """Inclusion-exclusion over every subset: the volume of a union of boxes, independent of the slicing."""
    inside = [p for p in points if all(x < r for x, r in zip(p, ref))]
    total = 0.0
    for size in range(1, len(inside) + 1):
        for subset in itertools.combinations(inside, size):
            corner = [max(p[j] for p in subset) for j in range(len(ref))]
            total += (-1) ** (size + 1) * math.prod(r - c for r, c in zip(ref, corner))
    return total


# --- cases calculated by hand --------------------------------------------------------------


def test_one_point_is_one_box() -> None:
    # (1 - 0.2) * (1 - 0.5) * (2 - 1) * (4 - 2) = 0.8 * 0.5 * 1 * 2
    assert hypervolume([(0.2, 0.5, 1.0, 2.0)], (1.0, 1.0, 2.0, 4.0)) == pytest.approx(0.8)
    assert hypervolume([(0.5,)], (2.0,)) == pytest.approx(1.5)  # one objective: a length
    assert hypervolume([(3.0,), (1.0,), (2.0,)], (4.0,)) == pytest.approx(3.0)


def test_two_dimensions_is_the_area_of_a_staircase() -> None:
    # [1,4]x[3,4] = 3 and [2,4]x[1,4] = 6 overlap in [2,4]x[3,4] = 2: 3 + 6 - 2
    assert hypervolume([(1.0, 3.0), (2.0, 1.0)], (4.0, 4.0)) == pytest.approx(7.0)
    # a third point under the staircase adds nothing, one on it adds exactly its notch
    assert hypervolume([(1.0, 3.0), (2.0, 1.0), (3.0, 3.5)], (4.0, 4.0)) == pytest.approx(7.0)
    assert hypervolume([(1.0, 3.0), (2.0, 1.0), (1.5, 2.0)], (4.0, 4.0)) == pytest.approx(7.0 + 0.5 * 1.0)


def test_three_dimensions_union_of_two_boxes() -> None:
    # A = (0,0,1): 3*3*2 = 18; B = (1,1,0): 2*2*3 = 12; both: [1,3]x[1,3]x[1,3] = 8; union 22
    assert hypervolume([(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)], (3.0, 3.0, 3.0)) == pytest.approx(22.0)


def test_four_dimensions_union_of_two_boxes_and_of_three() -> None:
    ref = (2.0, 2.0, 2.0, 2.0)
    # A = (0,0,0,1): 2*2*2*1 = 8; B = (1,1,1,0): 1*1*1*2 = 2; both: [1,2]^4 = 1; union 9
    assert hypervolume([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0)], ref) == pytest.approx(9.0)
    # C = (0,1,1,0): 2*1*1*2 = 4. C and A: [0,2]x[1,2]x[1,2]x[1,2] = 2; C and B: [1,2]^3 x [0,2] -> 1*1*1*2 = 2
    # (B inside C in the first three coordinates); all three: [1,2]^4 = 1. 8 + 2 + 4 - 2 - 1 - 2 + 1 = 10
    assert hypervolume([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0)], ref) == pytest.approx(10.0)


def test_dominated_duplicated_and_reordered_points_change_nothing() -> None:
    ref = (2.0, 2.0, 2.0, 2.0)
    front = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0)]
    base = hypervolume(front, ref)
    assert base == pytest.approx(9.0)
    assert hypervolume([*front, (0.5, 0.5, 0.5, 1.5)], ref) == base  # dominated by the first point
    assert hypervolume([*front, front[0], front[1]], ref) == base  # duplicates
    assert hypervolume(list(reversed(front)), ref) == base  # order


def test_a_point_at_or_beyond_the_reference_in_any_coordinate_adds_no_volume() -> None:
    ref = (2.0, 2.0, 2.0, 2.0)
    base = hypervolume([(0.0, 0.0, 0.0, 1.0)], ref)
    assert base == pytest.approx(8.0)
    for outside in ((3.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0), (0.0, 5.0, 0.0, 0.0)):
        assert hypervolume([(0.0, 0.0, 0.0, 1.0), outside], ref) == base  # even though it is "best" elsewhere
    assert hypervolume([], ref) == 0.0 and hypervolume([(9.0, 9.0, 9.0, 9.0)], ref) == 0.0


def test_scaling_every_coordinate_scales_the_volume_by_the_power_of_the_dimension() -> None:
    points = [(0.1, 0.4, 0.2, 0.3), (0.3, 0.1, 0.4, 0.2), (0.2, 0.2, 0.2, 0.2)]
    ref = (1.0, 1.0, 1.0, 1.0)
    base = hypervolume(points, ref)
    scaled = hypervolume([tuple(10 * x for x in p) for p in points], tuple(10 * r for r in ref))
    assert scaled == pytest.approx(base * 10**4)


# --- against an independent oracle ----------------------------------------------------------


@pytest.mark.parametrize("dims", [1, 2, 3, 4, 5])
def test_matches_inclusion_exclusion_on_random_fronts(dims: int) -> None:
    rng = random.Random(20261005 + dims)
    for _ in range(25):
        n = rng.randint(1, 8)
        # a coarse grid makes ties in single coordinates likely, the case slicing must get right
        points = [tuple(rng.randint(0, 9) / 10 for _ in range(dims)) for _ in range(n)]
        ref = tuple(rng.choice([0.8, 1.0, 1.2]) for _ in range(dims))
        assert hypervolume(points, ref) == pytest.approx(oracle(points, ref), rel=1e-9, abs=1e-12), (points, ref)


def test_adding_a_point_never_lowers_the_volume_and_a_big_front_is_fast() -> None:
    rng = random.Random(7)
    ref = (1.0, 1.0, 1.0, 1.0)
    points: list[tuple[float, ...]] = []
    last = 0.0
    for _ in range(40):
        points.append(tuple(rng.random() for _ in range(4)))
        value = hypervolume(points, ref)
        assert value >= last - 1e-12
        last = value
    # 300 points on a 4-D trade-off surface: all of them non-dominated
    surface = []
    for i in range(300):
        a, b, c = rng.random(), rng.random(), rng.random()
        total = a + b + c + 1.0
        surface.append((a / total, b / total, c / total, 1.0 / total))
    assert 0.0 < hypervolume(surface, (1.0, 1.0, 1.0, 1.0)) < 1.0


# --- keyed vectors, outside points and the protocol's reference ----------------------------


def test_hypervolume_of_lists_the_candidates_beyond_the_reference_point() -> None:
    ref = (1.0, 2.0, 4.0, 4.0)
    result = hypervolume_of(
        {
            "run.g000.c0002": (0.2, 1.0, 2.0, 2.0),
            "run.g000.c0001": (0.5, 0.5, 3.0, 3.0),
            "run.g000.c0000": (0.9, 2.5, 1.0, 1.0),  # latency beyond the point
            "run.g000.c0003": (1.0, 0.1, 0.1, 0.1),  # error exactly at the point: no width, listed
            "run.g000.c0004": None,  # no valid measurement: ignored, not listed
            "run.g000.c0005": (0.6, 1.5, 3.5, 3.5),  # dominated by c0001: inside the box, but adds nothing
        },
        ref,
    )
    assert isinstance(result, HypervolumeResult) and result.reference == ref
    assert result.outside == ("run.g000.c0000", "run.g000.c0003")  # sorted by id
    assert result.n_points == 5 and result.n_front == 2  # c0002 and c0001 shape the volume
    raw = hypervolume([(0.2, 1.0, 2.0, 2.0), (0.5, 0.5, 3.0, 3.0)], ref)
    assert result.raw == pytest.approx(raw) and result.scale == math.prod(ref)
    assert result.value == pytest.approx(raw / math.prod(ref))  # the official value is normalized
    # the volume of the whole set equals that of its non-dominated part
    assert result.raw == pytest.approx(oracle([(0.2, 1.0, 2.0, 2.0), (0.5, 0.5, 3.0, 3.0)], ref))
    assert hypervolume_of({}, ref) == HypervolumeResult(0.0, ref, 0, 0, ())
    assert hypervolume_of({"x": None}, ref).value == 0.0


def test_the_official_value_divides_every_objective_by_its_reference_coordinate() -> None:
    ref = (1.0, 2.5, 12.5, 5.0)
    points = [(0.2, 1.0, 5.0, 2.0), (0.5, 0.5, 8.0, 1.0), (0.1, 2.0, 11.0, 4.0)]
    raw = hypervolume(points, ref)
    value = normalized_hypervolume(points, ref)
    assert value == pytest.approx(raw / (1.0 * 2.5 * 12.5 * 5.0)) and 0.0 < value < 1.0
    # F~_j = F_j / r_j: the same number as the raw volume of the divided points against (1, 1, 1, 1)
    divided = [tuple(x / r for x, r in zip(p, ref)) for p in points]
    assert value == pytest.approx(hypervolume(divided, (1.0, 1.0, 1.0, 1.0)), rel=1e-12)
    assert hypervolume_of({str(i): p for i, p in enumerate(points)}, ref).value == pytest.approx(value)
    # a point at the origin dominates the whole box: 1; nothing inside the box: 0
    assert normalized_hypervolume([(0.0, 0.0, 0.0, 0.0)], ref) == pytest.approx(1.0)
    assert normalized_hypervolume([], ref) == 0.0 and normalized_hypervolume([(1.0, 9.0, 9.0, 9.0)], ref) == 0.0
    # the units of L, V and S do not matter: measuring V in kB instead of MB changes the raw volume, not the value
    rescaled = [(e, l, v * 1000.0, s) for e, l, v, s in points]
    assert hypervolume(rescaled, (1.0, 2.5, 12_500.0, 5.0)) == pytest.approx(raw * 1000.0)
    assert normalized_hypervolume(rescaled, (1.0, 2.5, 12_500.0, 5.0)) == pytest.approx(value)
    # the larger reference point of the sensitivity analysis gives a value on the same scale
    assert 0.0 < normalized_hypervolume(points, sensitivity_reference(ref)) <= 1.0
    for bad in ((1.0, 0.0, 1.0, 1.0), (1.0, -1.0, 1.0, 1.0)):
        with pytest.raises(ValueError, match="positive"):
            normalized_hypervolume([(0.1, 0.1, 0.1, 0.1)], bad)
        with pytest.raises(ValueError, match="positive"):
            hypervolume_of({"a": (0.1, 0.1, 0.1, 0.1)}, bad)


def test_reference_point_of_the_protocol_and_its_sensitivity() -> None:
    pilot = [(0.5, 1.0, 10.0, 2.0), (0.4, 2.0, 8.0, 4.0), (0.7, 1.5, 12.0, 3.0)]
    ref = reference_from_pilot(pilot)
    assert ref == (1.0, 2.5, 15.0, 5.0)  # r_E = 1; 1.25 * max of L, V, S = 2, 12, 4
    assert sensitivity_reference(ref) == pytest.approx((1.0, 3.75, 22.5, 7.5))  # 1.5 * r_j for L, V, S
    assert reference_from_pilot(pilot, factor=2.0)[2] == 24.0
    with pytest.raises(ValueError, match="at least one"):
        reference_from_pilot([])
    with pytest.raises(ValueError, match="expected 4"):
        reference_from_pilot([(1.0, 2.0)])


def test_malformed_input_is_refused_not_guessed() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        hypervolume([(1.0, 2.0, 3.0)], (4.0, 4.0))
    with pytest.raises(ValueError, match="non-finite"):
        hypervolume([(float("nan"), 1.0)], (4.0, 4.0))
    with pytest.raises(ValueError, match="non-finite"):
        hypervolume([(1.0, 1.0)], (4.0, float("inf")))
    with pytest.raises(ValueError, match="at least one"):
        hypervolume([], ())
