"""The sentinel network as seen by the operation: when to measure it, what to record, what counts as drift (task A9).

TCC-01 section 3 re-measures one fixed network every 50 evaluations to watch the measurement itself. The
measurement (building the sentinel engine and benchmarking it) is science and lives in
``model_pipeline/sentinel.py``; the operation never imports it. It receives two functions, :class:`SentinelHooks`,
from the wiring, and decides *when* to call them and *what to write down*:

* **When.** A baseline before the first attempt of the run (or, for a run that began without one, at the first
  opportunity), then each time the run has 50 more *evaluated attempts*. An evaluated attempt is one that went
  through the evaluator: it carries the ``operation`` block that the metered evaluator adds, which is what keeps
  cache hits and repaired-invalid candidates out of the count, and the count survives a resume because it is read
  from the database. The hook runs **between attempts** (where the runner asks whether to stop), never during one,
  because it uses the GPU.
* **What.** Every measurement is a job event ``sentinel_measured`` (the whole measurement, its duration and the
  number of evaluated attempts when it was taken). The first measurement of a run is its reference. A later one that
  the drift function marks as drifted also writes ``sentinel_drift`` with the stretch of attempts it marks, and a
  measurement that fails or cannot be compared with the reference writes ``sentinel_failed``. A failed measurement
  never stops the job, and nothing already recorded is replaced or removed: the analysis labels the stretch.
* **Which stretch.** A drift is seen only when it is measured. The attempts evaluated since the previous
  measurement (this one excluded from the previous, included in this) are the ones the drift marks, because the
  drift may have begun anywhere inside that window. A drift that began and ended inside one window is not seen.
* **Cost.** The seconds a measurement took go in its event and are *not* part of the run's GPU budget: the sentinel
  is a control, not a search, and counting it would charge the methods for something they did not choose. The
  analysis sums and declares the total per run.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_EVERY = 50
DRIFT_LIMIT = 0.05
Measure = Callable[[], Mapping[str, Any]]
Drift = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
Record = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class SentinelHooks:
    """What the wiring gives the operation: the measurement and the comparison with the reference.

    ``drift(reference, current)`` returns at least ``{"drifted": bool, "l_rel": float, "v_rel": float, "limit": float}``
    and raises ``ValueError`` when the two are not comparable (a different engine).
    """

    measure: Measure
    drift: Drift
    every: int = DEFAULT_EVERY

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("the sentinel period must be a whole number of evaluated attempts >= 1")


@dataclass(slots=True)
class SentinelState:
    """Where the sentinel of one run stands, rebuilt from the events already recorded."""

    reference: Mapping[str, Any] | None = None
    last_count: int | None = None  # evaluated attempts when the last measurement (or failure) was taken
    next_at: int = 0
    measurements: int = 0
    log: list[str] = field(default_factory=list)  # what happened in this session, for tests and the worker log

    @classmethod
    def from_events(cls, events: Sequence[Mapping[str, Any]], every: int) -> SentinelState:
        """Rebuild from the sentinel events of every job of the run (oldest first)."""
        state = cls()
        for event in events:
            detail = event["detail"]
            count = int(detail.get("evaluated_attempts", 0))
            if event["kind"] == "sentinel_measured":
                state.measurements += 1
                if state.reference is None and detail.get("measurement"):
                    state.reference = detail["measurement"]
            if event["kind"] in ("sentinel_measured", "sentinel_failed"):
                state.last_count = count
        if state.last_count is not None:
            state.next_at = (state.last_count // every + 1) * every
        return state


def tick(
    hooks: SentinelHooks,
    state: SentinelState,
    count: int,
    attempt_ids: Callable[[], Sequence[str]],
    record: Record,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Measure if the run has reached the next point; True if a measurement was attempted.

    ``count`` is the number of evaluated attempts now; ``attempt_ids()`` lists them in commit order (only
    asked for when a drift has to name its stretch).
    """
    if count < state.next_at:
        return False
    previous = state.last_count
    state.next_at = (count // hooks.every + 1) * hooks.every  # whatever happens, the next try is one window later
    started = clock()
    try:
        measurement = dict(hooks.measure())
    except Exception as exc:  # a failed control must not stop the science
        record(
            "sentinel_failed",
            {
                "evaluated_attempts": count,
                "seconds": clock() - started,
                "stage": "measure",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            },
        )
        state.last_count = count
        state.log.append("failed")
        return True
    seconds = clock() - started
    first = state.reference is None
    record(
        "sentinel_measured",
        {
            "evaluated_attempts": count,
            "seconds": seconds,
            "measurement": measurement,
            "reference": first,
            "index": state.measurements + 1,
        },
    )
    state.measurements += 1
    state.last_count = count
    state.log.append("measured")
    if first:
        state.reference = measurement
        return True
    try:
        verdict = dict(hooks.drift(state.reference, measurement))  # type: ignore[arg-type]
    except ValueError as exc:  # not the same engine: the numbers cannot be compared, and that must be visible
        record(
            "sentinel_failed",
            {
                "evaluated_attempts": count,
                "seconds": 0.0,
                "stage": "compare",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            },
        )
        state.log.append("not comparable")
        return True
    if verdict.get("drifted"):
        start = (previous if previous is not None else 0) + 1
        ids = list(attempt_ids())[start - 1 : count]
        record(
            "sentinel_drift",
            {
                "evaluated_attempts": count,
                "evaluated_attempts_from": start,
                "evaluated_attempts_to": count,
                "attempt_ids": ids,
                "drift": verdict,
            },
        )
        state.log.append("drift")
    return True
