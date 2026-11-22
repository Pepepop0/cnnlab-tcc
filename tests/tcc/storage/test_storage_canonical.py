from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from cnnlab.tcc.contracts import EvaluationResult, Fitness
from cnnlab.tcc.storage import SerializationError
from cnnlab.tcc.storage.canonical import (
    canonical_json,
    format_utc,
    result_from_payload,
    result_to_payload,
)


def test_canonical_json_is_order_and_whitespace_free() -> None:
    assert canonical_json({"b": 1, "a": [1, (2, 3)], "c": "é"}) == '{"a":[1,[2,3]],"b":1,"c":"é"}'


@pytest.mark.parametrize("bad", [math.nan, math.inf, {1: "int key"}, {"x": object()}, {"s": {1, 2}}])
def test_non_canonical_values_are_rejected(bad: object) -> None:
    with pytest.raises(SerializationError):
        canonical_json(bad)


def test_int_and_float_fitness_hash_identically() -> None:
    a = EvaluationResult("r", "c", "a", "valid", Fitness(0.2, 1, 3, 2), 5)
    b = EvaluationResult("r", "c", "a", "valid", Fitness(0.2, 1.0, 3.0, 2.0), 5.0)
    assert canonical_json(result_to_payload(a)) == canonical_json(result_to_payload(b))


def test_result_round_trip() -> None:
    original = EvaluationResult(
        "r",
        "c",
        "a",
        "invalid",
        None,
        1.25,
        durations_s={"train": 1.0},
        quality={"curve": [0.9, 0.8]},
        failure_stage="export",
        failure_code="onnx_error",
        failure_detail="op not supported",
    )
    assert result_from_payload(result_to_payload(original)) == original


def test_timestamps_are_utc_with_z_and_never_naive() -> None:
    local = datetime(2026, 9, 27, 21, 30, tzinfo=timezone(timedelta(hours=-3)))
    assert format_utc(local) == "2026-09-28T00:30:00.000000Z"
    with pytest.raises(ValueError):
        format_utc(datetime(2026, 9, 27, 21, 30))
