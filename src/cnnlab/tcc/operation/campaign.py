"""The plan of the confirmatory campaign: generate it, freeze it in SQLite, export it, queue it (task A6).

One plan covers the whole campaign (TCC-01 sections 4 and 5, TCC-04): eight paired blocks of seeds,
the four methods, the order in which the 32 runs go, and the operating rules that must exist, frozen,
before the first confirmatory run: how the budget is cut, what an interruption costs, the numeric
tolerances and the code that runs. Every run of the campaign is created against this one frozen plan
(``DemoRunSpec.plan_id``), so the database refuses any run whose method, seeds or budget are not the
ones written here, and a frozen plan can never be edited (an amendment is a new plan that links to it).

What is decided here and what is not
------------------------------------
* The **seeds** come from a master seed by a stated rule (SHA-256), so anyone can re-derive them. A block
  is shared by the four methods (the comparison is paired): same search, training, calibration and
  retraining seeds. Seeds are distinct between blocks.
* The **order** rotates the methods through the positions of each block (a cyclic Latin square), so with
  eight blocks and four methods each method is first, second, third and fourth exactly twice. The order
  is part of the plan and the queue refuses to reorder it.
* The **cut rule** and the **interruption policy** describe what the operation service does today; the
  tests keep the two in step.
* **Pending fields** are the ones whose value comes from the pilot: the hypervolume reference point
  (TCC-01 section 5), the longest training time of one attempt, and two measurement tolerances. They are
  part of the plan from the start so nobody forgets them, but a plan with a pending field **cannot be
  frozen** (a frozen plan cannot be completed later): ``freeze`` refuses unless ``allow_pending`` is
  given, which only the tests of this module do.
* The values the pilot decides are **read from its report** (``tools/piloto/analisa.py`` writes
  ``piloto.json``), not typed: the epochs of the search, the reference point, the longest training time
  and the latency repetitions. The plan records the file's SHA-256 and the decisions it took from it, and
  refuses a settings file that disagrees with the report on any of them.
* The longest training time of an attempt is **enforced by the evaluator** (``EvaluatorConfig.train_time_limit_s``):
  the plan carries it as ``max_attempt_train_s`` and also as ``factory_args.train_time_limit_s``, the key the science
  wiring reads, so it reaches the evaluator through the same arguments the run is checked against. A run whose
  arguments differ from the plan's, or whose limit differs from ``max_attempt_train_s``, is refused.

The search space (``space_hash``), the dataset and the code revision are recorded too, and a run refuses
to start if any of them differs (``operation.demo.prepare_demo``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cnnlab.tcc.contracts import Method
from cnnlab.tcc.operation import builder_config, schedule
from cnnlab.tcc.operation.demo import PROTOCOL_DOC, DemoRunSpec, DemoWiring
from cnnlab.tcc.operation.provenance import CodeVersion, tags_at_head
from cnnlab.tcc.storage.canonical import canonical_sha256, sha256_hex
from cnnlab.tcc.storage.errors import ConsistencyError, NotFoundError, StorageError
from cnnlab.tcc.storage.plans import CONFIRMATORY_BLOCKS, PlanExport
from cnnlab.tcc.storage.readmodel import open_read_model
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

CAMPAIGN_FORMAT = "tcc-campaign-plan/1"
PURPOSE = "confirmatory-campaign"
DEFAULT_METHODS: tuple[Method, ...] = ("joint", "mono", "random", "sequential")
SEED_KINDS = ("search_seed", "train_seed", "calibration_seed", "retrain_seed")
SEED_LIMIT = 2**31 - 1  # the queue and the database take seeds up to this value
PLAN_EXPORT_NAME = "plano-campanha.json"
PLAN_HASH_NAME = "plano-campanha.sha256"
CODE_LABEL_PREFIX = "campanha"
# Science arguments every campaign run gets unless the settings say otherwise. The plan builds its wiring (dataset,
# space, evaluator label) from the same merge, so the plan cannot record the 100-per-class subset of the wiring's own
# defaults while the runs train on the full 45k split (the plan frozen on 2026-10-11 08:14 did, and its runs were refused).
RUN_DEFAULT_ARGS: dict[str, Any] = {"population": 30, "generations": None, "train_per_class": None}
# Defaults of the two measurement tolerances, fixed before the pilot: the CV of the latency medians is the
# protocol's (above 2 % the benchmark goes to 3,000 repetitions and, if it stays unstable, the runs stop); the protocol only
# says "material deviation" for the audit of V, so it takes the same 5 % as the drift of the sentinel network.
LATENCY_CV_MAX = 0.02
V_AUDIT_REL = 0.05

# What the operation service does at the budget and at an interruption. Each statement is held by a test of the
# code it describes (the budget cut and the crossing attempt: test_operation_service.py; new attempt ids and the kept
# time of an interrupted attempt: test_operation_demo.py; resuming by run id and the refusal on other code:
# test_operation_resume_run.py; the stages of the sequential method: test_science_sequential.py; the single retry and
# the exhausted search: test_operation_campaign.py and test_science_search.py). Change one and the other must follow.
CUT_RULE: dict[str, Any] = {
    "unit": "active GPU seconds of the run (training, pruning, QAT, export, build, measurement and failed attempts)",
    "checked": "between attempts, never inside one",
    "attempt_in_progress": "finishes and counts",
    "excess_over_budget": "stays in the ledger, is reported per run and is never removed or hidden",
    "idle_time": "time between sessions and machine downtime do not consume the budget",
    "sequential": "each stage ends between attempts at its own limit (half of the budget each); "
    "time a stage does not use is not given to the other, and the run ends at the total budget",
    "search_ends_before_the_budget": "a search that can no longer evaluate anything new ends by itself "
    "(event search_exhausted) and the unused time is declared",
    "budget_change": "forbidden after any result is seen (TCC-01 section 4)",
}
INTERRUPTION_POLICY: dict[str, Any] = {
    "attempt_ids": "a resumed run gives every retried candidate a new attempt id; the interrupted attempt is kept",
    "interrupted_attempt_time": "counts in the budget (work already spent by an attempt that failed consumes it)",
    "infrastructure_retries": "an attempt the evaluator reports as interrupted is retried once; the next "
    "interruption of the same candidate blocks the search with a saved checkpoint",
    "resume": "cnnlab-tcc run --run-id RUN; the stored spec is reused and the run is never derived again",
    "code_change_on_resume": "refused unless --allow-code-change, and then written in the claim event of the job",
    "lease": "one job at a time holds the GPU lease; an expired lease marks the job interrupted",
    "unsupervised_session_max_h": 48,
    "block_wall_clock_target_h": 24,
}


def derive_seed(master_seed: int, kind: str, block: int, attempt: int = 0) -> int:
    """The seed of ``kind`` for ``block``: SHA-256 of a fixed string, reduced to the range of the database."""
    text = f"cnnlab-tcc-campaign:{master_seed}:{kind}:{block}:{attempt}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % SEED_LIMIT


def derive_blocks(master_seed: int, n_blocks: int = CONFIRMATORY_BLOCKS) -> list[dict[str, int]]:
    """``n_blocks`` blocks, every seed of every block distinct from every other (a collision is re-drawn)."""
    taken: set[int] = set()
    blocks: list[dict[str, int]] = []
    for block in range(n_blocks):
        entry = {"seed_block": block}
        for kind in SEED_KINDS:
            attempt = 0
            while (seed := derive_seed(master_seed, kind, block, attempt)) in taken:
                attempt += 1
            taken.add(seed)
            entry[kind] = seed
        blocks.append(entry)
    return blocks


def run_order(n_blocks: int, methods: Sequence[str]) -> list[dict[str, Any]]:
    """Blocks in order; inside block ``b`` the methods start at position ``b mod n_methods`` (cyclic Latin square)."""
    order: list[dict[str, Any]] = []
    for block in range(n_blocks):
        shift = block % len(methods)
        for method in [*methods[shift:], *methods[:shift]]:
            order.append({"position": len(order) + 1, "seed_block": block, "method": method})
    return order


def position_counts(order: Sequence[Mapping[str, Any]], n_methods: int) -> dict[str, list[int]]:
    """For every method, how many times it ran first, second... inside its block (balanced when equal)."""
    counts: dict[str, list[int]] = {}
    for index, cell in enumerate(order):
        counts.setdefault(str(cell["method"]), [0] * n_methods)[index % n_methods] += 1
    return counts


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    """What the operator decides; everything else in the plan follows from it or from the wiring.

    ``factory_args`` are the science arguments of every run (population, epochs, slice, generations and the
    training data). ``reference_point``, ``max_attempt_train_s`` and the two tolerances are the pilot's.
    """

    master_seed: int
    factory: str
    factory_args: Mapping[str, Any]
    budget_gpu_s: float = 19_800.0
    methods: tuple[Method, ...] = DEFAULT_METHODS
    n_blocks: int = CONFIRMATORY_BLOCKS
    n_frozen: int = 7
    stage_gpu_s: tuple[float, float] | None = None
    reference_point: tuple[float, float, float, float] | None = None
    viable_minima: tuple[float, float, float, float] | None = None
    max_attempt_train_s: float | None = None
    pilot_report: str | None = None  # path of piloto.json; its decisions fill the fields above
    timing_cache: Mapping[str, Any] | None = (
        None  # the builder's TensorRT timing cache, if adopted (see builder_config)
    )
    sentinel_every_evaluations: int = 50
    sentinel_drift_rel: float = 0.05
    latency_cv_max: float | None = LATENCY_CV_MAX
    v_audit_rel: float | None = V_AUDIT_REL
    author: str = "Pedro Antunes"

    def __post_init__(self) -> None:
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int):
            raise ValueError("master_seed must be an int")
        if not self.factory or ":" not in self.factory:
            raise ValueError("factory must be 'package.module:function'")
        if not (
            isinstance(self.budget_gpu_s, (int, float))
            and self.budget_gpu_s > 0
            and float(self.budget_gpu_s) == int(self.budget_gpu_s)
        ):
            raise ValueError("budget_gpu_s must be a positive whole number of seconds (the queue takes integers)")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be distinct values")
        if "slice" not in self.factory_args:
            raise ValueError("factory_args.slice is required: the admitted slice the campaign searches")
        if "sequential" in self.methods and self.factory_args.get("generations") is not None:
            raise ValueError("the sequential method has no generation limit: factory_args.generations must be null")
        if self.n_blocks != CONFIRMATORY_BLOCKS:
            raise ValueError(
                f"n_blocks must be {CONFIRMATORY_BLOCKS}: the database refuses a confirmatory plan of another size"
            )
        if self.reference_point is not None:
            point = tuple(float(x) for x in self.reference_point)
            if len(point) != 4 or not all(math.isfinite(x) and x > 0 for x in point):
                raise ValueError("reference_point must be four finite positive numbers (E, L, V, S)")
            object.__setattr__(self, "reference_point", point)
        if self.viable_minima is not None:
            minima = tuple(float(x) for x in self.viable_minima)
            if len(minima) != 4 or not all(math.isfinite(x) and x >= 0 for x in minima):
                raise ValueError("viable_minima must be four finite non-negative numbers (E, L, V, S)")
            if self.reference_point is not None and not all(m < r for m, r in zip(minima, self.reference_point)):
                raise ValueError("every viable minimum must be below its coordinate of the reference point")
            object.__setattr__(self, "viable_minima", minima)
        for name in ("max_attempt_train_s", "latency_cv_max", "v_audit_rel"):
            value = getattr(self, name)
            if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number or null (pending the pilot)")
        if self.stage_gpu_s is not None:
            object.__setattr__(self, "stage_gpu_s", tuple(float(x) for x in self.stage_gpu_s))
        object.__setattr__(self, "timing_cache", builder_config.normalize(self.timing_cache))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CampaignSettings:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown campaign settings: {sorted(unknown)}")
        values = dict(data)
        for name in ("methods", "reference_point", "viable_minima", "stage_gpu_s"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> CampaignSettings:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8-sig")))


@dataclass(frozen=True, slots=True)
class PilotDecisions:
    """What the pilot report decided, as the plan needs it, and where it came from."""

    file_sha256: str
    run_id: str | None
    epochs: int | None
    reference_point: tuple[float, float, float, float] | None
    viable_minima: tuple[float, float, float, float] | None
    max_attempt_train_s: float | None
    latency_repetitions: int | None
    timing_cache: Mapping[str, Any] | None = None  # the builder's timing cache, when the pilot produced one

    def provenance(self) -> dict[str, Any]:
        return {
            "report_sha256": self.file_sha256,
            "run_id": self.run_id,
            "search_epochs": self.epochs,
            "latency_repetitions": self.latency_repetitions,
            "max_attempt_train_s": self.max_attempt_train_s,
            "reference_point": None if self.reference_point is None else list(self.reference_point),
            "viable_minima": None if self.viable_minima is None else list(self.viable_minima),
            "timing_cache": None if self.timing_cache is None else dict(self.timing_cache),
        }


def load_pilot(path: Path) -> PilotDecisions:
    """The decisions of ``piloto.json``. A decision the report could not take (``null``) stays ``None``."""
    raw = Path(path).read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    decisions = report.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError(f"{path}: not a pilot report (no 'decisions')")
    point = decisions.get("reference_point")
    reference = None
    if point is not None:
        reference = (float(point["E"]), float(point["L_ms"]), float(point["V_mb"]), float(point["S_mb"]))
    low = report.get("viable_minima")  # the pilot's minima of the viable candidates, next to its decisions
    minima = None if low is None else (float(low["E"]), float(low["L_ms"]), float(low["V_mb"]), float(low["S_mb"]))
    return PilotDecisions(
        file_sha256=sha256_hex(raw),
        run_id=report.get("run_id"),
        epochs=decisions.get("search_epochs"),
        reference_point=reference,
        viable_minima=minima,
        max_attempt_train_s=decisions.get("train_time_limit_s"),
        latency_repetitions=decisions.get("latency_repetitions"),
        timing_cache=builder_config.normalize(decisions.get("timing_cache")),
    )


def apply_pilot(settings: CampaignSettings, pilot: PilotDecisions) -> CampaignSettings:
    """The settings with the pilot's decisions filled in; a value typed in the settings must agree with the report."""
    args = dict(settings.factory_args)
    changes: dict[str, Any] = {}

    def fill(name: str, typed: Any, decided: Any, same: Any = lambda a, b: a == b) -> Any:
        if decided is None:
            return typed
        if typed is not None and not same(typed, decided):
            raise ConsistencyError(f"{name} is {typed!r} in the settings but the pilot decided {decided!r}")
        return decided

    args["epochs"] = fill("epochs", args.get("epochs"), pilot.epochs)
    args["bench_iters"] = fill("bench_iters (latency repetitions)", args.get("bench_iters"), pilot.latency_repetitions)
    changes["reference_point"] = fill(
        "reference_point",
        settings.reference_point,
        pilot.reference_point,
        lambda a, b: all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(a, b)),
    )
    changes["viable_minima"] = fill(
        "viable_minima",
        settings.viable_minima,
        pilot.viable_minima,
        lambda a, b: all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(a, b)),
    )
    changes["max_attempt_train_s"] = fill(
        "max_attempt_train_s", settings.max_attempt_train_s, pilot.max_attempt_train_s
    )
    typed_cache, decided_cache = settings.timing_cache, pilot.timing_cache
    if decided_cache is not None:
        if typed_cache is not None and (
            typed_cache["adopted"] != decided_cache["adopted"]
            or (typed_cache["path"] is not None and typed_cache != decided_cache)
        ):
            raise ConsistencyError(
                f"timing_cache is {typed_cache!r} in the settings but the pilot decided {decided_cache!r}"
            )
        changes["timing_cache"] = decided_cache
    return replace(settings, factory_args={k: v for k, v in args.items() if v is not None}, **changes)


def pending_fields(content: Mapping[str, Any]) -> list[str]:
    """Names of the plan fields that are still ``null`` and come from the pilot."""
    out = []
    if content.get("reference_point") is None:
        out.append("reference_point")
    if content.get("viable_minima") is None:
        out.append("viable_minima")
    if content.get("max_attempt_train_s") is None:
        out.append("max_attempt_train_s")
    for name, value in dict(content.get("tolerances", {})).items():
        if value is None:
            out.append(f"tolerances.{name}")
    out += builder_config.pending((content.get("builder") or {}).get("timing_cache"))
    if "epochs" not in content.get("factory_args", {}):
        out.append("factory_args.epochs")
    return out


def code_label(plan_content: Mapping[str, Any]) -> str:
    """The label every run of the campaign carries, so all of them share one environment."""
    return f"{CODE_LABEL_PREFIX}-{plan_content['code_freeze']['git_sha'][:8]}"


def build_content(
    settings: CampaignSettings,
    wiring: DemoWiring,
    code_version: CodeVersion,
    protocol_id: str,
    pilot: PilotDecisions | None = None,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    """The plan content (JSON-native, deterministic) for these settings, this wiring and this code."""
    if not code_version.is_known:
        raise ConsistencyError(f"code version unknown ({code_version.error}): the plan fixes the code that runs")
    blocks = derive_blocks(settings.master_seed, settings.n_blocks)
    order = run_order(settings.n_blocks, settings.methods)
    factory_args = {**RUN_DEFAULT_ARGS, **settings.factory_args}
    if settings.reference_point is not None:
        factory_args["reference_point"] = list(settings.reference_point)
    factory_args["sentinel_every"] = settings.sentinel_every_evaluations  # the plan asks for the sentinel network ...
    factory_args["sentinel_drift_limit"] = settings.sentinel_drift_rel  # ... and says how much drift is too much
    if settings.max_attempt_train_s is not None:
        typed = factory_args.get("train_time_limit_s")
        if typed is not None and float(typed) != float(settings.max_attempt_train_s):
            raise ConsistencyError(
                f"factory_args.train_time_limit_s is {typed!r} but max_attempt_train_s is {settings.max_attempt_train_s!r}: "
                "they are one limit; give it once"
            )
        factory_args["train_time_limit_s"] = float(settings.max_attempt_train_s)  # what the evaluator reads
    if "sequential" in settings.methods:
        factory_args["n_frozen"] = settings.n_frozen
        if settings.stage_gpu_s is not None:
            factory_args["stage_gpu_s"] = list(settings.stage_gpu_s)
    content: dict[str, Any] = {
        "format": CAMPAIGN_FORMAT,
        "purpose": PURPOSE,
        "stage_scope": "confirmatory",
        "confirmatory": True,
        "note": "Confirmatory campaign: every run of every method and block is created against this frozen plan.",
        "blocks": blocks,
        "methods": list(settings.methods),
        "budget_gpu_s": float(settings.budget_gpu_s),
        "run_order": order,
        "order_rule": "blocks in order; inside block b the methods start at position b mod n_methods, "
        "in the cyclic order of `methods` (a Latin square: each method takes every position equally)",
        "position_counts": position_counts(order, len(settings.methods)),
        "seed_rule": {
            "master_seed": settings.master_seed,
            "derivation": "int(sha256('cnnlab-tcc-campaign:<master>:<kind>:<block>:<attempt>')[:12], 16) % (2**31 - 1); "
            "a collision between any two seeds of the plan raises <attempt>",
            "kinds": list(SEED_KINDS),
            "paired": "the four methods of a block share the same four seeds",
        },
        "dataset_id": wiring.dataset.dataset_id,
        "space_hash": wiring.space_hash,
        "evaluator": wiring.evaluator_label,
        "factory": settings.factory,
        "factory_args": factory_args,
        "protocol_id": protocol_id,
        "code_freeze": {
            "git_sha": code_version.git_sha,
            "dirty": bool(code_version.dirty),
            "dirty_fingerprint": code_version.dirty_fingerprint,
            "label": f"{CODE_LABEL_PREFIX}-{str(code_version.git_sha)[:8]}",
            "tags": sorted(tags),  # the git tags on the frozen commit, as the operator named the freeze
        },
        "reference_point": None if settings.reference_point is None else list(settings.reference_point),
        "viable_minima": None if settings.viable_minima is None else list(settings.viable_minima),
        "viable_minima_rule": "minima of E, L, V, S over the viable candidates of the pilot; with the reference point they "
        "normalize the balance role of the finalists, (f - m) / (r - m) (TCC-01 section 6)",
        "reference_rule": "r_E = 1 and r_j = 1.25 x max_j of the viable pilot genomes for L, V, S (TCC-01 section 5); "
        "the hypervolume is normalized by it (each objective divided by its coordinate); "
        "the sensitivity analysis uses r'_j = 1.5 x r_j",
        "pilot": None if pilot is None else pilot.provenance(),
        "max_attempt_train_s": settings.max_attempt_train_s,
        "max_attempt_train_s_rule": "longest training time of one attempt; the value comes from the pilot and reaches the "
        "evaluator as factory_args.train_time_limit_s, which stops a longer training "
        "(EvaluatorConfig.train_time_limit_s)",
        "builder": {
            "timing_cache": None if settings.timing_cache is None else dict(settings.timing_cache),
            "timing_cache_rule": builder_config.RULE,
        },
        "cut_rule": CUT_RULE,
        "interruption_policy": INTERRUPTION_POLICY,
        "tolerances": {
            "sentinel_every_evaluations": settings.sentinel_every_evaluations,
            "sentinel_drift_rel": settings.sentinel_drift_rel,
            "sentinel_rule": "a sentinel network every N evaluations; a drift above the tolerance opens a new "
            "measurement linked to the original (task G6)",
            "latency_cv_max": settings.latency_cv_max,
            "v_audit_rel": settings.v_audit_rel,
            "latency_cv_rule": "CV of the medians of L over 10 sessions; above the limit the benchmark goes to 3,000 repetitions and, if it stays unstable, the runs stop (TCC-01 section 5)",
            "v_audit_rule": "relative deviation of V against an allocation of known size; the protocol says only 'material deviation', so the limit is the sentinel drift (5 %), fixed by the coordinator before the pilot",
        },
    }
    if "sequential" in settings.methods:
        content["sequential"] = {
            "n_frozen": settings.n_frozen,
            "stage_gpu_s": None if settings.stage_gpu_s is None else list(settings.stage_gpu_s),
            "split": "half of the run budget per stage unless stage_gpu_s is given (2.75 h + 2.75 h)",
        }
    content["pending"] = pending_fields(content)
    return content


def representative_spec(settings: CampaignSettings) -> DemoRunSpec:
    """A spec of the first block, joint method: enough to ask the wiring factory for the dataset, the space and the labels."""
    block = derive_blocks(settings.master_seed, 1)[0]
    return DemoRunSpec(
        method="joint",
        seed_block=0,
        search_seed=block["search_seed"],
        train_seed=block["train_seed"],
        calibration_seed=block["calibration_seed"],
        budget_gpu_s=float(settings.budget_gpu_s),
        code_label=CODE_LABEL_PREFIX,
        author=settings.author,
        factory=settings.factory,
        factory_args={**RUN_DEFAULT_ARGS, **settings.factory_args},
    )


def preview(
    settings: CampaignSettings,
    wiring: DemoWiring,
    code_version: CodeVersion,
    repo_root: Path,
    pilot: PilotDecisions | None = None,
) -> tuple[str, dict[str, Any]]:
    """(plan id, content) as :func:`create_plan` would write them, without touching any database."""
    doc = Path(repo_root) / PROTOCOL_DOC
    if not doc.is_file():
        raise NotFoundError(f"protocol document not found: {doc}")
    protocol_id = f"tcc01-{sha256_hex(doc.read_bytes())[:12]}"
    if pilot is not None:
        settings = apply_pilot(settings, pilot)
    content = build_content(settings, wiring, code_version, protocol_id, pilot, tags_at_head(repo_root))
    return plan_id_of(content), content


def plan_id_of(content: Mapping[str, Any]) -> str:
    return f"campaign-plan-{canonical_sha256(content)[:12]}"


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    plan_id: str
    plan_sha256: str
    status: str
    content: Mapping[str, Any]
    pending: tuple[str, ...] = field(default=())

    @property
    def frozen(self) -> bool:
        return self.status == "frozen"


def create_plan(
    store: SQLiteRunStore,
    settings: CampaignSettings,
    wiring: DemoWiring,
    code_version: CodeVersion,
    repo_root: Path,
    *,
    freeze: bool = False,
    allow_pending: bool = False,
    allow_dirty: bool = False,
    pilot: PilotDecisions | None = None,
) -> CampaignPlan:
    """Write the plan as a draft, or freeze it; safe to repeat (the id comes from the content).

    A frozen plan is complete: it is refused while a field of the pilot is still pending and when the
    code that will run is not a clean revision.
    """
    doc = Path(repo_root) / PROTOCOL_DOC
    if not doc.is_file():
        raise NotFoundError(f"protocol document not found: {doc}")
    digest = sha256_hex(doc.read_bytes())
    protocol_id = f"tcc01-{digest[:12]}"
    if pilot is not None:
        settings = apply_pilot(settings, pilot)
    content = build_content(settings, wiring, code_version, protocol_id, pilot, tags_at_head(repo_root))
    pending = tuple(content["pending"])
    builder_config.verify(
        content["builder"]["timing_cache"], repo_root
    )  # a named cache must be there, with those bytes
    if freeze:
        if pending and not allow_pending:
            raise ConsistencyError(
                "the plan cannot be frozen while these fields wait for the pilot: "
                + ", ".join(pending)
                + " (a frozen plan cannot be completed; fill them in the settings and generate it again)"
            )
        if code_version.dirty and not allow_dirty:
            raise ConsistencyError(
                "the plan cannot be frozen on a dirty tree: the campaign runs one committed revision"
            )
    store.register_protocol(protocol_id, digest, "confirmed")
    plan_id = plan_id_of(content)
    try:
        record = store.get_plan(plan_id)
    except NotFoundError:
        store.create_plan(plan_id, protocol_id, content, author=settings.author)
        record = store.get_plan(plan_id)
    if canonical_sha256(record.content) != canonical_sha256(content):
        raise ConsistencyError(f"plan {plan_id!r} exists with other content")
    if freeze and record.status != "frozen":
        record = store.freeze_plan(plan_id)
    return CampaignPlan(plan_id, record.plan_sha256, record.status, dict(record.content), pending)


def attempt_time_limit(plan_content: Mapping[str, Any]) -> float | None:
    """Seconds of training one attempt may take (None: the plan sets none). For the evaluator to read."""
    value = plan_content.get("max_attempt_train_s")
    return None if value is None else float(value)


# -- export (read-only) --------------------------------------------------------------------------


def read_plan(db_path: Path, plan_id: str) -> dict[str, Any]:
    """The plan as stored, through the read model; the stored hash is checked against its content."""
    with open_read_model(Path(db_path)) as rm:
        return rm.plan(plan_id)


def export_plan(db_path: Path, plan_id: str, dest: Path) -> PlanExport:
    """Write the plan body and its SHA-256 into the new directory ``dest``; never overwrites, never writes the database."""
    dest = Path(dest)
    if dest.exists():
        raise StorageError(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    plan = read_plan(db_path, plan_id)
    body = plan["body"].encode("utf-8")
    if sha256_hex(body) != plan["plan_sha256"]:
        raise ConsistencyError(f"plan {plan_id!r} hash does not match its content")
    dest.mkdir(parents=True)
    (dest / PLAN_EXPORT_NAME).write_bytes(body)
    (dest / PLAN_HASH_NAME).write_text(f"{plan['plan_sha256']}  {PLAN_EXPORT_NAME}\n", encoding="utf-8")
    return PlanExport(plan_id, plan["status"], plan["frozen_at"], body, plan["plan_sha256"])


# -- queue ----------------------------------------------------------------------------------------


def queue_items(plan: Mapping[str, Any]) -> list[tuple[str, schedule.RunSettings]]:
    """(name, settings) of every run of the plan, in the order of the plan."""
    content = plan["content"]
    args = dict(content["factory_args"])
    blocks = {b["seed_block"]: b for b in content["blocks"]}
    budget = content["budget_gpu_s"]
    if float(budget) != int(budget):
        raise ConsistencyError("the queue takes whole seconds of budget")
    form_keys = {"train_per_class", "epochs", "population", "generations", "slice"}
    extra = {k: v for k, v in args.items() if k not in form_keys}  # what the form has no field for
    items = []
    for cell in content["run_order"]:
        block = blocks[cell["seed_block"]]
        settings = schedule.RunSettings(
            slice=args.get("slice", "fp32"),
            population=args["population"],
            generations=args.get("generations"),
            budget_gpu_s=int(budget),
            epochs=args["epochs"],
            train_per_class=args.get("train_per_class"),
            search_seed=block["search_seed"],
            train_seed=block["train_seed"],
            method=cell["method"],
            calibration_seed=block["calibration_seed"],
            seed_block=cell["seed_block"],
            plan_id=plan["plan_id"],
            factory_extra=extra,
            code_label=content["code_freeze"]["label"],
            factory=content["factory"],
        )
        items.append((f"{cell['method']}-b{cell['seed_block']}", settings))
    return items


def enqueue(queue_dir: Path, plan: Mapping[str, Any]) -> list[str]:
    """Add the runs of a frozen plan to the queue in plan order; cells already in the queue are skipped.

    Returns the names added. The queue refuses to edit or reorder these items afterwards.
    """
    if plan["status"] != "frozen":
        raise ConsistencyError(f"plan {plan['plan_id']!r} is {plan['status']}: only a frozen plan can be queued")
    wanted = queue_items(plan)
    position = {(s.method, s.seed_block): n for n, (_, s) in enumerate(wanted)}
    added: list[str] = []

    def change(state: schedule.QueueState) -> schedule.QueueState:
        present = {(i.settings.plan_id, i.settings.method, i.settings.seed_block) for i in state.items}
        last = max((position[(m, b)] for p, m, b in present if p == plan["plan_id"]), default=-1)
        for name, settings in wanted:
            if (settings.plan_id, settings.method, settings.seed_block) in present:
                continue
            if position[(settings.method, settings.seed_block)] < last:
                raise schedule.ScheduleError(
                    f"{name} would go to the end of the queue after runs that come later in the plan: "
                    "delete the later runs of this plan first, so the order of the plan holds"
                )
            state = schedule.add_item(state, name, settings, tag=schedule.CAMPAIGN)
            added.append(name)
        return state

    schedule.mutate(Path(queue_dir), change)
    return added


def summary(plan: Mapping[str, Any]) -> str:
    """A few lines for the terminal: identity, what is fixed, what is pending."""
    content = plan["content"]
    lines = [
        f"plan {plan['plan_id']} [{plan['status']}] sha256={plan['plan_sha256']}",
        f"  {len(content['blocks'])} blocks x {len(content['methods'])} methods = {len(content['run_order'])} runs, "
        f"budget {content['budget_gpu_s']:.0f} s of active GPU each",
        f"  code {content['code_freeze']['git_sha'][:12]} | space {content['space_hash'][:12]} | dataset {content['dataset_id']}",
        "  order: " + " ".join(f"b{c['seed_block']}:{c['method'][:3]}" for c in content["run_order"][:8]) + " ...",
    ]
    pending = pending_fields(content)
    lines.append("  pending (from the pilot): " + (", ".join(pending) if pending else "none"))
    return "\n".join(lines)
