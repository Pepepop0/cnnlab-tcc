"""Search space v1 of the TCC genome (docs/TCC-02-GENOMA.md sections 2, 3 and 6).

The base space contains exactly the closed domains of TCC-02 section 2. The
conditional techniques (FP8, INT4/FP4 weight-only, 2:4 sparsity) exist only as
*candidates*: a :class:`SearchSpace` refuses to include one unless it receives
an :class:`AdmissionEvidence` that satisfies criteria (a), (b) and (c) of the
admission test. The default :data:`SPACE_V1` has no admitted conditional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from cnnlab.tcc.science.hashing import digest, is_sha256_hex

GENOME_SCHEMA = "tcc-genome/v1"
EXPRESSED_SCHEMA = "tcc-genome-expressed/v1"
SPACE_SCHEMA = "tcc-space/v1"
REPAIR_RULES_VERSION = "tcc-repair/v1"

N_BLOCK_SLOTS = 6
INPUT_RESOLUTION = 32
MAX_SPATIAL_REDUCTIONS = 4
CHANNEL_ROUNDING = 16
GROUPED_CONV_GROUPS = 4
NUM_CLASSES = 100
HEAD_HIDDEN_UNITS = 256

PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8")
OPS: tuple[str, ...] = ("standard", "grouped_g4", "dw_separable", "mbconv_e3", "mbconv_e6")
DOWNSAMPLES: tuple[str, ...] = ("none", "maxpool", "avgpool", "conv_s2")
INT8_GENES: tuple[str, ...] = ("int8_mode", "weight_granularity", "calibration")
MBCONV_EXPANSION = {"mbconv_e3": 3, "mbconv_e6": 6}


@dataclass(frozen=True, slots=True)
class Choice:
    """Finite domain. ``ordinal`` domains are ordered for one-step mutation."""

    values: tuple[Any, ...]
    kind: Literal["ordinal", "categorical"]

    def contains(self, value: object) -> bool:
        # Exact type match: True must not pass as 1, nor 16.0 as 16.
        return any(type(value) is type(v) and value == v for v in self.values)

    def definition(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class Interval:
    """Closed real interval; ``log10_exponent`` genes already live in log space."""

    low: float
    high: float
    scale: Literal["log10_exponent", "linear"]

    @property
    def kind(self) -> str:
        return "continuous"

    def contains(self, value: object) -> bool:
        return (
            isinstance(value, float)
            and value == value  # rejects NaN
            and self.low <= value <= self.high
        )

    def definition(self) -> dict[str, Any]:
        return {"kind": "continuous", "low": self.low, "high": self.high, "scale": self.scale}


Domain = Union[Choice, Interval]

_BASE_GLOBAL_DOMAINS: tuple[tuple[str, Domain], ...] = (
    ("n_blocks", Choice((2, 3, 4, 5, 6), "ordinal")),
    ("stem_channels", Choice((16, 32, 64), "ordinal")),
    ("activation", Choice(("relu", "silu", "hardswish"), "categorical")),
    ("head", Choice(("gap_linear", "gap_fc256_linear"), "categorical")),
    ("stem_precision", Choice(PRECISIONS, "categorical")),
    ("head_precision", Choice(PRECISIONS, "categorical")),
    ("int8_mode", Choice(("ptq", "qat"), "categorical")),
    ("weight_granularity", Choice(("per_tensor", "per_channel"), "categorical")),
    ("calibration", Choice(("max", "entropy", "percentile_99_99"), "categorical")),
    ("log10_lr", Interval(-4.0, -2.0, "log10_exponent")),
    ("log10_wd", Interval(-6.0, -3.0, "log10_exponent")),
    ("head_dropout", Interval(0.0, 0.5, "linear")),
)

_BASE_BLOCK_DOMAINS: tuple[tuple[str, Domain], ...] = (
    ("channels", Choice((16, 32, 48, 64, 96, 128, 192, 256), "ordinal")),
    ("op", Choice(OPS, "categorical")),
    ("kernel", Choice((3, 5, 7), "ordinal")),
    ("n_convs", Choice((1, 2, 3), "ordinal")),
    ("residual", Choice((False, True), "categorical")),
    ("downsample", Choice(DOWNSAMPLES, "categorical")),
    ("precision", Choice(PRECISIONS, "categorical")),
    ("prune_pct", Choice((0, 25, 50), "ordinal")),
)

GLOBAL_GENES: tuple[str, ...] = tuple(name for name, _ in _BASE_GLOBAL_DOMAINS)
BLOCK_GENES: tuple[str, ...] = tuple(name for name, _ in _BASE_BLOCK_DOMAINS)
CONTINUOUS_GENES: tuple[str, ...] = tuple(name for name, dom in _BASE_GLOBAL_DOMAINS if isinstance(dom, Interval))

# --- conditional techniques (TCC-02 section 2, last row, and section 6) -----

ConditionalTechnique = Literal["fp8", "int4_weights", "fp4_weights", "sparse_2_4"]
CONDITIONAL_TECHNIQUES: tuple[str, ...] = ("fp8", "int4_weights", "fp4_weights", "sparse_2_4")
CONDITIONAL_PRECISION_VALUE: dict[str, str] = {
    "fp8": "fp8",
    "int4_weights": "int4w",
    "fp4_weights": "fp4w",
}
# Deterministic repair target when a block's op leaves the admitted scope.
PRECISION_FALLBACK: dict[str, str] = {"fp8": "fp16", "int4w": "int8", "fp4w": "int8"}
SPARSITY_VALUES: tuple[str, ...] = ("dense", "2:4")
SCOPE_TARGETS: tuple[str, ...] = OPS + ("stem", "head")
# TCC-02 rule 5: plain depthwise has one channel per group, never 4.
SPARSE_FORBIDDEN_TARGETS: tuple[str, ...] = ("dw_separable", "stem", "head")
REPRESENTATIVE_ARCHITECTURES = 3
MIN_ARCHITECTURES_WITH_GAIN = 2


class AdmissionError(ValueError):
    """A conditional technique was offered to the space without passing admission."""


@dataclass(frozen=True, slots=True)
class AdmissionEvidence:
    """Outcome of the TCC-02 section 6 admission test for one conditional technique.

    Negative outcomes are representable (and should be tabulated); only a
    ``passed`` evidence can extend a :class:`SearchSpace`.
    """

    technique: str
    scope: tuple[str, ...]
    evidence_sha256: str
    builds_and_runs: bool  # criterion (a), for every target in ``scope``
    format_confirmed_in_engine: bool  # criterion (b), via engine inspector
    architectures_with_gain: int  # criterion (c): L/V gain > 2x sentinel noise, or S by bytes
    architectures_failed: int = 0
    representative_architectures: int = REPRESENTATIVE_ARCHITECTURES

    def __post_init__(self) -> None:
        if self.technique not in CONDITIONAL_TECHNIQUES:
            raise AdmissionError(f"unknown conditional technique {self.technique!r}")
        scope = tuple(sorted(set(self.scope)))
        if not scope or any(t not in SCOPE_TARGETS for t in scope):
            raise AdmissionError(f"scope must be a non-empty subset of {SCOPE_TARGETS}")
        if self.technique == "sparse_2_4" and set(scope) & set(SPARSE_FORBIDDEN_TARGETS):
            raise AdmissionError("2:4 sparsity cannot be declared for depthwise, stem or head")
        if not is_sha256_hex(self.evidence_sha256):
            raise AdmissionError("evidence_sha256 must reference the admission report by SHA-256")
        if self.representative_architectures != REPRESENTATIVE_ARCHITECTURES:
            raise AdmissionError("the admission test uses exactly three representative architectures")
        counts = (self.architectures_with_gain, self.architectures_failed)
        if any(type(c) is not int or c < 0 for c in counts) or sum(counts) > REPRESENTATIVE_ARCHITECTURES:
            raise AdmissionError("architecture counts must be non-negative and sum to at most 3")
        object.__setattr__(self, "scope", scope)

    @property
    def passed(self) -> bool:
        """Criteria (a), (b) and (c) of TCC-02 section 6 all hold."""
        return (
            self.builds_and_runs
            and self.format_confirmed_in_engine
            and self.architectures_with_gain >= MIN_ARCHITECTURES_WITH_GAIN
        )

    def definition(self) -> dict[str, Any]:
        return {
            "technique": self.technique,
            "scope": list(self.scope),
            "evidence_sha256": self.evidence_sha256,
            "builds_and_runs": self.builds_and_runs,
            "format_confirmed_in_engine": self.format_confirmed_in_engine,
            "architectures_with_gain": self.architectures_with_gain,
            "architectures_failed": self.architectures_failed,
            "representative_architectures": self.representative_architectures,
        }


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Domains in force for one run. Its :meth:`space_hash` versions the whole space."""

    admissions: tuple[AdmissionEvidence, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for evidence in self.admissions:
            if not isinstance(evidence, AdmissionEvidence):
                raise AdmissionError("admissions must be AdmissionEvidence objects")
            if not evidence.passed:
                raise AdmissionError(f"{evidence.technique} did not pass the admission test and cannot enter the space")
            if evidence.technique in seen:
                raise AdmissionError(f"duplicate admission for {evidence.technique}")
            seen.add(evidence.technique)
        ordered = tuple(sorted(self.admissions, key=lambda e: e.technique))
        object.__setattr__(self, "admissions", ordered)

    # -- conditional helpers ------------------------------------------------

    def admitted_scope(self, technique: str) -> tuple[str, ...]:
        for evidence in self.admissions:
            if evidence.technique == technique:
                return evidence.scope
        return ()

    @property
    def has_sparsity_gene(self) -> bool:
        return bool(self.admitted_scope("sparse_2_4"))

    def precision_values(self, target: str) -> tuple[str, ...]:
        """Conditionally valid precisions for an op, ``"stem"`` or ``"head"``."""
        if target not in SCOPE_TARGETS:
            raise ValueError(f"unknown precision target {target!r}")
        extra = tuple(
            value
            for technique, value in CONDITIONAL_PRECISION_VALUE.items()
            if target in self.admitted_scope(technique)
        )
        return PRECISIONS + extra

    def block_precision_values(self) -> tuple[str, ...]:
        """Raw-genome domain of a block precision: base plus any admitted for some op."""
        extra = tuple(
            value
            for technique, value in CONDITIONAL_PRECISION_VALUE.items()
            if set(self.admitted_scope(technique)) & set(OPS)
        )
        return PRECISIONS + extra

    def sparsity_values(self, op: str) -> tuple[str, ...]:
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}")
        if op in self.admitted_scope("sparse_2_4"):
            return SPARSITY_VALUES
        return ("dense",)

    # -- domains ------------------------------------------------------------

    def global_domains(self) -> tuple[tuple[str, Domain], ...]:
        out: list[tuple[str, Domain]] = []
        for name, domain in _BASE_GLOBAL_DOMAINS:
            if name == "stem_precision":
                domain = Choice(self.precision_values("stem"), "categorical")
            elif name == "head_precision":
                domain = Choice(self.precision_values("head"), "categorical")
            out.append((name, domain))
        return tuple(out)

    def block_domains(self) -> tuple[tuple[str, Domain], ...]:
        out: list[tuple[str, Domain]] = []
        for name, domain in _BASE_BLOCK_DOMAINS:
            if name == "precision":
                domain = Choice(self.block_precision_values(), "categorical")
            out.append((name, domain))
        if self.has_sparsity_gene:
            out.append(("sparsity", Choice(SPARSITY_VALUES, "categorical")))
        return tuple(out)

    def definition(self) -> dict[str, Any]:
        """Everything that changes the set of reachable phenotypes or their meaning."""
        return {
            "schema": SPACE_SCHEMA,
            "genome_schema": GENOME_SCHEMA,
            "expressed_schema": EXPRESSED_SCHEMA,
            "repair_rules": REPAIR_RULES_VERSION,
            "constants": {
                "n_block_slots": N_BLOCK_SLOTS,
                "input_resolution": INPUT_RESOLUTION,
                "max_spatial_reductions": MAX_SPATIAL_REDUCTIONS,
                "channel_rounding": CHANNEL_ROUNDING,
                "grouped_conv_groups": GROUPED_CONV_GROUPS,
                "num_classes": NUM_CLASSES,
                "head_hidden_units": HEAD_HIDDEN_UNITS,
                "mbconv_expansion": dict(MBCONV_EXPANSION),
            },
            "global": {name: dom.definition() for name, dom in self.global_domains()},
            "block": {name: dom.definition() for name, dom in self.block_domains()},
            "precision_fallback": {k: v for k, v in PRECISION_FALLBACK.items() if k in self.block_precision_values()},
            "admissions": [e.definition() for e in self.admissions],
        }

    def space_hash(self) -> str:
        return digest("tcc.space/v1", self.definition())


SPACE_V1 = SearchSpace()
"""Base space: TCC-02 domains only, no conditional technique admitted."""
