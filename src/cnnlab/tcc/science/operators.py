"""Genetic operators on the raw genome and the FP32/unpruned search slice (TCC-02 section 4).

Operators work on :class:`Genome` (all six slots, latent ones included), are
pure given a :class:`GeneRng`, and always return genomes whose genes lie in
their domains; the deterministic repair of :func:`express` then handles the
active part (e.g. more than four reductions).

* Crossover: whole block slots and global genes are taken from either parent
  with probability 1/2 each (uniform crossover by block and by global gene).
* Mutation: each gene mutates with probability ``1/n_genes`` and at least one
  gene always changes. Ordinal genes move one step, categorical genes are
  resampled among the *other* values, continuous genes get a Gaussian step
  of ``sigma_rel`` times the interval width, reflected at the bounds (the
  log-rate genes already live in log10 space).

Representation slices
---------------------
A :class:`SearchSlice` is a declared sub-space: some genes pinned to one
value, others restricted to a subset of their domain. Each slice has its own
hash, so its results can never be mistaken for results of the full
``SPACE_V1`` or of another slice.

* :data:`FP32_SLICE` pins ``stem_precision``, ``head_precision`` and every
  block ``precision`` to ``fp32`` and every ``prune_pct`` to 0.
* :data:`FP16_BLOCK_SLICE` keeps stem and head in ``fp32`` and pruning at 0,
  and lets each block choose its precision in ``{fp32, fp16}``.
* :data:`FP16_REGIONS_SLICE` also lets the stem and the head choose in
  ``{fp32, fp16}`` (pruning still 0).

* :data:`ADMITTED_SLICE` and :data:`ARCHITECTURE_SLICE` are the two
  slices of the campaign, defined at the end of this module: the admitted
  space (everything the pipeline implements and the engine runs) and its
  FP32 architecture restriction.

Precision scopes
----------------
A representation that the engine runs only for some operations or activations
is *conditioned* to them (TCC-02 section 6). A :class:`PrecisionScope` states
where a precision may be used; it is part of the slice definition and so of its
hash. Inside the slice the scope is not a repair applied afterwards: sampling
draws a region's precision only among the values valid for its operation and
activation, and :meth:`SearchSlice.pin` (called after every mutation and
crossover) moves a precision that the scope forbids to the scope's fallback,
so every genome the operators return is inside the slice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from cnnlab.tcc.contracts import GenomeRecord
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import BlockGenes, Genome, validate_genome
from cnnlab.tcc.science.hashing import digest
from cnnlab.tcc.science.record import CacheContext, PreparedGenome, cache_context_hash, phenotype_hash
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1, Choice, Interval, SearchSpace

OPERATORS_VERSION = "tcc-operators/v1"
DEFAULT_CROSSOVER_P = 0.6
DEFAULT_MUTATION_P = 0.3
DEFAULT_SIGMA_REL = 0.15
MAX_SCOPE_RETRIES = 8


@dataclass(frozen=True, slots=True)
class PrecisionScope:
    """Where a precision may run, as measured in the engine: some operations and activations only.

    ``block_*`` apply to a block (its operation and the activation of the network); ``stem_activations`` to the
    stem; ``head_hidden_activations`` to a head with a hidden layer (``gap_fc256_linear``; the plain linear head
    has no activation, so nothing limits it). A region outside the scope takes ``fallback`` instead.
    ``evidence`` names the measurement the scope comes from.
    """

    precision: str
    block_ops: tuple[str, ...]
    block_activations: tuple[str, ...]
    stem_activations: tuple[str, ...]
    head_hidden_activations: tuple[str, ...]
    fallback: str
    evidence: str

    def allows(self, region: str, op: str | None, activation: str, head: str | None = None) -> bool:
        if region == "stem":
            return activation in self.stem_activations
        if region == "head":
            return head != "gap_fc256_linear" or activation in self.head_hidden_activations
        return op in self.block_ops and activation in self.block_activations

    def definition(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "block_ops": list(self.block_ops),
            "block_activations": list(self.block_activations),
            "stem_activations": list(self.stem_activations),
            "head_hidden_activations": list(self.head_hidden_activations),
            "fallback": self.fallback,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class SearchSlice:
    """A base space with genes pinned to one value or restricted to a subset (a declared sub-space).

    ``allowed_*`` hold ``(gene, values)`` with the values in the order of the
    base domain. A slice without restrictions has the same definition, hence
    the same hash, as before restrictions existed.
    """

    base: SearchSpace
    pinned_global: tuple[tuple[str, Any], ...]
    pinned_block: tuple[tuple[str, Any], ...]
    reason: str
    allowed_global: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    allowed_block: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    scopes: tuple[PrecisionScope, ...] = ()

    def __post_init__(self) -> None:
        self._check_scopes()
        for kind, domains, pinned, allowed in (
            ("global", dict(self.base.global_domains()), dict(self.pinned_global), self.allowed_global),
            ("block", dict(self.base.block_domains()), dict(self.pinned_block), self.allowed_block),
        ):
            for name, value in pinned.items():
                if name not in domains or not domains[name].contains(value):
                    raise ValueError(f"pinned {kind} gene {name}={value!r} is outside the base space")
            seen: set[str] = set()
            for name, values in allowed:
                domain = domains.get(name)
                if not isinstance(domain, Choice):
                    raise ValueError(f"restricted {kind} gene {name!r} must be a finite gene of the base space")
                if name in pinned or name in seen:
                    raise ValueError(f"{kind} gene {name!r} is declared twice in the slice")
                seen.add(name)
                inside = tuple(v for v in domain.values if any(type(v) is type(a) and v == a for a in values))
                if len(values) < 2 or len(inside) != len(values) or inside != tuple(values):
                    raise ValueError(
                        f"restricted {kind} gene {name!r} needs at least two values of {domain.values}, "
                        f"in that order; got {values!r}"
                    )

    def _check_scopes(self) -> None:
        precisions = dict(self.base.global_domains())["stem_precision"].values
        activations = dict(self.base.global_domains())["activation"].values
        ops = dict(self.base.block_domains())["op"].values
        seen: set[str] = set()
        for scope in self.scopes:
            if scope.precision in seen or scope.precision not in precisions or scope.fallback not in precisions:
                raise ValueError(f"scope of {scope.precision!r}: unknown or repeated precision or fallback")
            seen.add(scope.precision)
            if scope.fallback == scope.precision:
                raise ValueError(f"scope of {scope.precision!r}: the fallback must differ from the precision")
            if not (
                set(scope.block_ops) <= set(ops)
                and set(scope.block_activations) <= set(activations)
                and set(scope.stem_activations) <= set(activations)
                and set(scope.head_hidden_activations) <= set(activations)
            ):
                raise ValueError(f"scope of {scope.precision!r}: operations or activations outside the base space")
            for kind in ("stem_precision", "head_precision"):
                if kind in dict(self.pinned_global) and dict(self.pinned_global)[kind] == scope.precision:
                    raise ValueError(f"scope of {scope.precision!r}: {kind} is pinned to the scoped precision")
            for kind, allowed in (
                ("stem_precision", dict(self.allowed_global).get("stem_precision")),
                ("head_precision", dict(self.allowed_global).get("head_precision")),
                ("precision", dict(self.allowed_block).get("precision")),
            ):
                if allowed is not None and scope.fallback not in allowed and scope.precision in allowed:
                    raise ValueError(f"scope of {scope.precision!r}: the fallback is not allowed for {kind}")

    def definition(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": "tcc-space-slice/v1",
            "base": self.base.definition(),
            "pinned_global": dict(self.pinned_global),
            "pinned_block": dict(self.pinned_block),
            "reason": self.reason,
        }
        if self.allowed_global:
            out["allowed_global"] = {name: list(values) for name, values in self.allowed_global}
        if self.allowed_block:
            out["allowed_block"] = {name: list(values) for name, values in self.allowed_block}
        if self.scopes:
            out["precision_scopes"] = [scope.definition() for scope in self.scopes]
        return out

    def space_hash(self) -> str:
        return digest("tcc.space-slice/v1", self.definition())

    def restricted_domains(self) -> dict[str, Choice]:
        """Domains of the restricted genes, keyed like frozen genes (``name`` or ``block.name``)."""
        base_global = dict(self.base.global_domains())
        base_block = dict(self.base.block_domains())
        out = {name: Choice(tuple(values), base_global[name].kind) for name, values in self.allowed_global}
        out.update(
            {f"block.{name}": Choice(tuple(values), base_block[name].kind) for name, values in self.allowed_block}
        )
        return out

    def _scope_of(self, precision: str) -> PrecisionScope | None:
        return next((s for s in self.scopes if s.precision == precision), None)

    def _admits(self, precision: str, region: str, op: str | None, activation: str, head: str | None = None) -> bool:
        scope = self._scope_of(precision)
        return scope is None or scope.allows(region, op, activation, head)

    def pin(self, genome: Genome) -> Genome:
        """Return ``genome`` with every pinned gene (latent slots included) set to its value.

        With precision scopes, a region whose precision its operation or activation does not allow takes the
        scope's fallback, so the result is always inside the slice (idempotent, and no random draw).
        """
        blocks = tuple(replace(b, **dict(self.pinned_block)) for b in genome.blocks)
        genome = replace(genome, **dict(self.pinned_global), blocks=blocks)
        if not self.scopes:
            return genome
        activation = genome.activation
        stem, head = genome.stem_precision, genome.head_precision
        if not self._admits(stem, "stem", None, activation):
            stem = self._scope_of(stem).fallback  # type: ignore[union-attr]
        if not self._admits(head, "head", None, activation, genome.head):
            head = self._scope_of(head).fallback  # type: ignore[union-attr]
        blocks = tuple(
            b
            if self._admits(b.precision, "block", b.op, activation)
            else replace(b, precision=self._scope_of(b.precision).fallback)  # type: ignore[union-attr]
            for b in genome.blocks
        )
        return replace(genome, stem_precision=stem, head_precision=head, blocks=blocks)

    def contains(self, genome: Genome) -> bool:
        """Pinned genes have their value and restricted genes a permitted one, latent slots included."""
        if self.pin(genome) != genome:
            return False
        domains = self.restricted_domains()
        if any(not domains[name].contains(getattr(genome, name)) for name, _ in self.allowed_global):
            return False  # (precision scopes are enforced by ``pin`` above)
        return all(
            domains[f"block.{name}"].contains(getattr(block, name))
            for name, _ in self.allowed_block
            for block in genome.blocks
        )

    def sample(self, rng: GeneRng) -> Genome:
        """Draw from the base space, pin, then draw each restricted gene uniformly from its subset.

        Pinned and restricted genes still consume their base draws, and every
        restricted gene takes exactly one more draw per slot, so the number
        of draws per genome is fixed.
        """
        genome = self.pin(sample_genome(self.base, rng))
        if self.allowed_global or self.allowed_block:
            globals_ = {name: rng.choice(values) for name, values in self.allowed_global}
            blocks = tuple(
                replace(b, **{name: rng.choice(values) for name, values in self.allowed_block}) for b in genome.blocks
            )
            genome = replace(genome, **globals_, blocks=blocks)
        return self._draw_scoped(genome, rng) if self.scopes else genome

    def _precision_values(self, gene: str, block: bool) -> tuple[str, ...]:
        """Values a region's precision may take in this slice (before the scopes)."""
        pinned = dict(self.pinned_block if block else self.pinned_global)
        if gene in pinned:
            return (pinned[gene],)
        allowed = dict(self.allowed_block if block else self.allowed_global)
        if gene in allowed:
            return tuple(allowed[gene])
        domains = dict(self.base.block_domains() if block else self.base.global_domains())
        return tuple(domains[gene].values)

    def _draw_scoped(self, genome: Genome, rng: GeneRng) -> Genome:
        """Redraw every free precision uniformly among the values its region's scope allows.

        Drawing among the valid values keeps the sample uniform over them (projecting a bad draw to the fallback
        would pile probability on the fallback). One draw per free region, in the order stem, slots, head.
        """
        activation = genome.activation
        stem_values, head_values = (
            self._precision_values("stem_precision", False),
            self._precision_values("head_precision", False),
        )
        block_values = self._precision_values("precision", True)
        free_stem, free_head = len(stem_values) > 1, len(head_values) > 1
        free_block = len(block_values) > 1
        stem = (
            rng.choice(tuple(v for v in stem_values if self._admits(v, "stem", None, activation)))
            if free_stem
            else genome.stem_precision
        )
        blocks = tuple(
            replace(
                b, precision=rng.choice(tuple(v for v in block_values if self._admits(v, "block", b.op, activation)))
            )
            if free_block
            else b
            for b in genome.blocks
        )
        head = (
            rng.choice(tuple(v for v in head_values if self._admits(v, "head", None, activation, genome.head)))
            if free_head
            else genome.head_precision
        )
        return self.pin(replace(genome, stem_precision=stem, head_precision=head, blocks=blocks))


FP32_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(("head_precision", "fp32"), ("stem_precision", "fp32")),
    pinned_block=(("precision", "fp32"), ("prune_pct", 0)),
    reason="pipeline implements FP32 unpruned engines only (no Q/DQ, per-region precision or physical pruning)",
)

FP16_BLOCK_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(("head_precision", "fp32"), ("stem_precision", "fp32")),
    pinned_block=(("prune_pct", 0),),
    reason="stem and head in FP32, no pruning; each block runs in FP32 or FP16 with casts at the boundaries "
    "(no Q/DQ, BF16 or physical pruning in the pipeline)",
    allowed_block=(("precision", ("fp32", "fp16")),),
)

FP16_REGIONS_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(),
    pinned_block=(("prune_pct", 0),),
    reason="no pruning; stem, every block and the head each run in FP32 or FP16 with casts at the boundaries "
    "(no Q/DQ, BF16 or physical pruning in the pipeline)",
    allowed_global=(("stem_precision", ("fp32", "fp16")), ("head_precision", ("fp32", "fp16"))),
    allowed_block=(("precision", ("fp32", "fp16")),),
)

# Slices that the pilot wiring can run end to end (the evaluator materializes FP16 in every region).
SLICES: dict[str, SearchSlice] = {
    "fp32": FP32_SLICE,
    "fp16_block": FP16_BLOCK_SLICE,
    "fp16_regions": FP16_REGIONS_SLICE,
}

BF16_REGIONS_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(),
    pinned_block=(("prune_pct", 0),),
    reason="no pruning; stem, every block and the head each run in FP32, FP16 or BF16 with casts at the "
    "boundaries (BF16 combinations the engine cannot run, such as HardSwish or depthwise, are invalid)",
    allowed_global=(("stem_precision", ("fp32", "fp16", "bf16")), ("head_precision", ("fp32", "fp16", "bf16"))),
    allowed_block=(("precision", ("fp32", "fp16", "bf16")),),
)
SLICES["bf16_regions"] = BF16_REGIONS_SLICE

INT8_PTQ_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(("int8_mode", "ptq"),),
    pinned_block=(("prune_pct", 0),),
    reason="no pruning; stem, every block and the head each run in FP32, FP16, BF16 or INT8 (post-training "
    "quantization with explicit Q/DQ, calibrated on the 1,024 frozen training examples); combinations the "
    "engine cannot run in the requested type are invalid",
    allowed_global=(
        ("stem_precision", ("fp32", "fp16", "bf16", "int8")),
        ("head_precision", ("fp32", "fp16", "bf16", "int8")),
    ),
    allowed_block=(("precision", ("fp32", "fp16", "bf16", "int8")),),
)
SLICES["int8_ptq"] = INT8_PTQ_SLICE

# Slices for the pruning task (G5): pilot use only, the admitted slices of the campaign are G7's.
PRUNE_FP32_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(("head_precision", "fp32"), ("stem_precision", "fp32")),
    pinned_block=(("precision", "fp32"),),
    reason="FP32 everywhere; every block prunes 0%, 25% or 50% of its channels physically, then fine-tunes one epoch",
)
SLICES["prune_fp32"] = PRUNE_FP32_SLICE

COMPRESSED_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(),
    pinned_block=(),
    reason="stem, blocks and head in FP32, FP16 or INT8 (PTQ or QAT) and per-block physical pruning (0%, 25%, 50%) "
    "with one fine-tune epoch; BF16 and sparsity are left out",
    allowed_global=(("stem_precision", ("fp32", "fp16", "int8")), ("head_precision", ("fp32", "fp16", "int8"))),
    allowed_block=(("precision", ("fp32", "fp16", "int8")),),
)
SLICES["compressed"] = COMPRESSED_SLICE

INT8_QAT_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(("int8_mode", "qat"),),
    pinned_block=(("prune_pct", 0),),
    reason="no pruning; stem, every block and the head each run in FP32, FP16, BF16 or INT8 (quantization-aware "
    "training: one fine-tune epoch with simulated Q/DQ after BN folding, ranges initialised on the 1,024 "
    "frozen training examples); combinations the engine cannot run in the requested type are invalid",
    allowed_global=(
        ("stem_precision", ("fp32", "fp16", "bf16", "int8")),
        ("head_precision", ("fp32", "fp16", "bf16", "int8")),
    ),
    allowed_block=(("precision", ("fp32", "fp16", "bf16", "int8")),),
)
SLICES["int8_qat"] = INT8_QAT_SLICE
# --- the slices of the campaign ---------------------------------------------------------------

BF16_SCOPE = PrecisionScope(
    precision="bf16",
    block_ops=("standard", "grouped_g4"),
    block_activations=("relu", "silu"),
    stem_activations=("relu", "silu"),
    head_hidden_activations=("relu", "silu"),
    fallback="fp16",
    evidence="TensorRT 11.3 on the RTX 5070 (precision_probe --matrix bf16, docs/implementation/science/wave3-gauss.md): "
    "HardSwish does not build in BF16 and depthwise convolutions (dw_separable, mbconv_e3, mbconv_e6) run as "
    "FP16 there; standard and grouped_g4 blocks, the stem and the head run in BF16 with ReLU or SiLU",
)

ADMITTED_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(),
    pinned_block=(),
    reason="admitted space of the campaign: everything the pipeline implements and the engine runs. Stem, blocks and "
    "head each run in FP32, FP16, BF16 or INT8 (PTQ or QAT, per-tensor or per-channel weights, max, entropy or "
    "99.99th-percentile calibration) and every block prunes 0%, 25% or 50% of its channels physically. BF16 is "
    "conditioned to the operations and activations the engine runs in BF16 (standard and grouped_g4 blocks "
    "with ReLU or SiLU; the stem with ReLU or SiLU; the head, with ReLU or SiLU when it has a hidden layer); "
    "elsewhere a region takes FP16. FP8, INT4, FP4 and 2:4 sparsity are not admitted (no admission evidence)",
    scopes=(BF16_SCOPE,),
)

ARCHITECTURE_SLICE = SearchSlice(
    base=SPACE_V1,
    pinned_global=(
        ("head_precision", "fp32"),
        ("stem_precision", "fp32"),
        ("int8_mode", "ptq"),
        ("weight_granularity", "per_channel"),
        ("calibration", "max"),
    ),
    pinned_block=(("precision", "fp32"), ("prune_pct", 0)),
    reason="first stage of the sequential method: the architecture and training genes vary, the representation is "
    "pinned to FP32 without pruning and the INT8 genes (not expressed without an INT8 region) to constants; "
    "every genome is inside ADMITTED_SLICE, whose base space it shares",
)

SLICES["admitted"] = ADMITTED_SLICE
SLICES["architecture"] = ARCHITECTURE_SLICE
PROBE_SLICES: dict[str, SearchSlice] = dict(SLICES)  # kept apart so the probe can take slices the wiring cannot run


def prepare_in_slice(genome: Genome, space: SearchSlice, context: CacheContext) -> PreparedGenome:
    """Like ``record.prepare_genome`` but hashed with the slice identity."""
    if not space.contains(genome):
        raise ValueError("genome has genes outside the slice")
    space_hash = space.space_hash()
    if context.space_hash != space_hash:
        raise ValueError("cache context was built for another search space")
    expression = express(genome, space.base)
    phenotype = phenotype_hash(expression.genome)
    record = GenomeRecord(
        raw_genome=genome.to_dict(),
        expressed_genome=expression.genome.to_dict(),
        repair_log=expression.repair_log,
        phenotype_hash=phenotype,
        cache_context_hash=cache_context_hash(phenotype, context),
        space_hash=space_hash,
    )
    return PreparedGenome(record=record, expression=expression)


# --- operators --------------------------------------------------------------------


def crossover(a: Genome, b: Genome, rng: GeneRng, space: SearchSpace = SPACE_V1) -> tuple[Genome, Genome]:
    """Uniform crossover by global gene and by whole block slot; returns two complementary children."""
    ga: dict[str, Any] = {}
    gb: dict[str, Any] = {}
    for name, _ in space.global_domains():
        x, y = getattr(a, name), getattr(b, name)
        if rng.unit() < 0.5:
            x, y = y, x
        ga[name], gb[name] = x, y
    blocks_a: list[BlockGenes] = []
    blocks_b: list[BlockGenes] = []
    for x, y in zip(a.blocks, b.blocks):
        if rng.unit() < 0.5:
            x, y = y, x
        blocks_a.append(x)
        blocks_b.append(y)
    return Genome(**ga, blocks=tuple(blocks_a)), Genome(**gb, blocks=tuple(blocks_b))


def _gauss(rng: GeneRng) -> float:
    """Standard normal from two uniforms (Box-Muller), so only ``unit()`` is consumed."""
    u1 = max(rng.unit(), 1e-12)
    u2 = rng.unit()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _reflect(value: float, low: float, high: float) -> float:
    width = high - low
    if width <= 0:
        return low
    offset = (value - low) % (2 * width)
    return low + (offset if offset <= width else 2 * width - offset)


def mutate_value(domain: Choice | Interval, value: Any, rng: GeneRng, sigma_rel: float = DEFAULT_SIGMA_REL) -> Any:
    """One mutation step of one gene; the result always differs for finite domains of size > 1."""
    if isinstance(domain, Interval):
        step = _gauss(rng) * sigma_rel * (domain.high - domain.low)
        return float(_reflect(float(value) + step, domain.low, domain.high))
    values = domain.values
    if len(values) < 2:
        return value
    index = next(i for i, v in enumerate(values) if type(v) is type(value) and v == value)
    if domain.kind == "ordinal":
        if index == 0:
            return values[1]
        if index == len(values) - 1:
            return values[-2]
        return values[index - 1] if rng.unit() < 0.5 else values[index + 1]
    others = [v for i, v in enumerate(values) if i != index]
    return rng.choice(others)


def _gene_paths(
    genome: Genome,
    space: SearchSpace,
    frozen: set[str],
    restricted: Mapping[str, Choice],
) -> list[tuple[str, int | None, Choice | Interval]]:
    """Mutable genes: globals plus genes of the *active* blocks (latent ones drift only by crossover).

    A restricted gene mutates inside its subset, not inside the base domain.
    """
    paths: list[tuple[str, int | None, Choice | Interval]] = [
        (name, None, restricted.get(name, dom)) for name, dom in space.global_domains() if name not in frozen
    ]
    for slot in range(genome.n_blocks):
        paths += [
            (name, slot, restricted.get(f"block.{name}", dom))
            for name, dom in space.block_domains()
            if f"block.{name}" not in frozen
        ]
    return paths


def mutate(
    genome: Genome,
    rng: GeneRng,
    space: SearchSpace = SPACE_V1,
    sigma_rel: float = DEFAULT_SIGMA_REL,
    frozen: frozenset[str] = frozenset(),
    restricted: Mapping[str, Choice] | None = None,
) -> Genome:
    """Mutate each eligible gene with probability 1/n; at least one gene changes.

    ``frozen`` holds gene names that must not move (global names, or
    ``"block.<name>"``), used by slices whose pinned genes are constant.
    ``restricted`` maps such names to the subset a gene may take.
    """
    paths = _gene_paths(genome, space, set(frozen), restricted or {})
    chosen = [p for p in paths if rng.unit() < 1.0 / len(paths)]
    if not chosen:
        chosen = [paths[rng.index(len(paths))]]
    globals_ = {name: getattr(genome, name) for name, _ in space.global_domains()}
    blocks = [b.to_dict() for b in genome.blocks]
    for name, slot, domain in chosen:
        if slot is None:
            globals_[name] = mutate_value(domain, globals_[name], rng, sigma_rel)
        else:
            blocks[slot][name] = mutate_value(domain, blocks[slot][name], rng, sigma_rel)
    child = Genome(**globals_, blocks=tuple(BlockGenes.from_dict(b) for b in blocks))
    validate_genome(child, space)
    return child


def slice_frozen_genes(space: SearchSlice) -> frozenset[str]:
    return frozenset([name for name, _ in space.pinned_global] + [f"block.{name}" for name, _ in space.pinned_block])


def make_offspring(
    a: Genome,
    b: Genome,
    rng: GeneRng,
    space: SearchSlice,
    crossover_p: float = DEFAULT_CROSSOVER_P,
    mutation_p: float = DEFAULT_MUTATION_P,
    sigma_rel: float = DEFAULT_SIGMA_REL,
) -> tuple[Genome, Genome, dict[str, Any]]:
    """Two children of ``a`` and ``b`` inside ``space``, with the operators applied to each.

    A child identical to a parent after crossover is always mutated, so
    variation never silently returns a copy.
    """
    frozen = slice_frozen_genes(space)
    restricted = space.restricted_domains()
    crossed = rng.unit() < crossover_p
    c1, c2 = crossover(a, b, rng, space.base) if crossed else (a, b)
    out: list[Genome] = []
    mutated: list[bool] = []
    for child in (c1, c2):
        do_mutate = rng.unit() < mutation_p or child == a or child == b
        if do_mutate:
            child = mutate(child, rng, space.base, sigma_rel, frozen, restricted)
        child = space.pin(child)
        if space.scopes:  # the projection onto the scopes can undo a mutation: mutate again rather than copy
            for _ in range(MAX_SCOPE_RETRIES):
                if child not in (a, b):
                    break
                child = space.pin(mutate(child, rng, space.base, sigma_rel, frozen, restricted))
        out.append(child)
        mutated.append(do_mutate)
    return out[0], out[1], {"crossover": crossed, "mutated": mutated, "operators": OPERATORS_VERSION}
