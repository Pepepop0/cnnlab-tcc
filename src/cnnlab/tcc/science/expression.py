"""Pure, deterministic repair and expression of a raw genome (TCC-02 section 3).

``express`` never mutates its input (both genome types are frozen) and uses no
randomness: the same raw genome and space always give the same expressed
genome and the same repair log. Only active blocks are repaired; latent slots
stay as proposed so that operators can still recombine them.

Rules applied, in order:

* ``C1_conditional_scope`` -- a conditional precision/sparsity value on a block
  whose op is outside the admitted scope falls back deterministically
  (reachable only once a conditional technique is admitted).
* ``R1_max_spatial_reductions`` -- at most four active reductions
  (32->16->8->4->2); the excess becomes ``none`` starting from the last block.
* ``R2`` grouped ``g=4`` divisibility and ``R3`` pruning rounding are checked or
  computed by :func:`plan_network`; with v1 domains (multiples of 16) R2 always
  holds, so it is an invariant check rather than a repair.
* Quantization genes are expressed only when some part runs INT8, and the PTQ
  calibration gene only in PTQ mode (TCC-02 rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from cnnlab.tcc.science.genome import BlockGenes, ExpressedGenome, Genome, validate_genome
from cnnlab.tcc.science.hashing import canonical_json
from cnnlab.tcc.science.space import (
    CHANNEL_ROUNDING,
    GROUPED_CONV_GROUPS,
    HEAD_HIDDEN_UNITS,
    INPUT_RESOLUTION,
    MAX_SPATIAL_REDUCTIONS,
    MBCONV_EXPANSION,
    NUM_CLASSES,
    PRECISION_FALLBACK,
    SearchSpace,
)

RULE_CONDITIONAL_SCOPE = "C1_conditional_scope"
RULE_MAX_REDUCTIONS = "R1_max_spatial_reductions"


@dataclass(frozen=True, slots=True)
class Repair:
    """One deterministic change from raw to expressed value."""

    rule: str
    path: str
    before: Any
    after: Any

    def to_log(self) -> str:
        """Canonical JSON line, parseable by storage and analysis."""
        return canonical_json({"rule": self.rule, "path": self.path, "before": self.before, "after": self.after})


@dataclass(frozen=True, slots=True)
class Expression:
    """Result of repairing and expressing one raw genome.

    ``invalid_reason`` is set when no deterministic repair preserves a buildable
    topology; such a candidate must be recorded as invalid without E/L/V/S.
    """

    genome: ExpressedGenome
    repairs: tuple[Repair, ...]
    invalid_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    @property
    def repair_log(self) -> tuple[str, ...]:
        return tuple(r.to_log() for r in self.repairs)


def express(genome: Genome, space: SearchSpace) -> Expression:
    """Repair the active part of ``genome`` and drop every non-expressed gene."""
    validate_genome(genome, space)
    repairs: list[Repair] = []
    active = [_repair_scope(i, b, space, repairs) for i, b in enumerate(genome.active_blocks)]
    active = _repair_reductions(active, repairs)

    precisions = (genome.stem_precision, genome.head_precision, *(b.precision for b in active))
    uses_int8 = "int8" in precisions
    expressed = ExpressedGenome(
        n_blocks=genome.n_blocks,
        stem_channels=genome.stem_channels,
        activation=genome.activation,
        head=genome.head,
        stem_precision=genome.stem_precision,
        head_precision=genome.head_precision,
        int8_mode=genome.int8_mode if uses_int8 else None,
        weight_granularity=genome.weight_granularity if uses_int8 else None,
        calibration=genome.calibration if uses_int8 and genome.int8_mode == "ptq" else None,
        log10_lr=genome.log10_lr,
        log10_wd=genome.log10_wd,
        head_dropout=genome.head_dropout,
        blocks=tuple(active),
    )
    return Expression(expressed, tuple(repairs), _topology_violation(plan_network(expressed)))


def repaired_genome(genome: Genome, space: SearchSpace) -> Genome:
    """Full-length raw genome with active slots replaced by their repaired values.

    Returns a new object; ``genome`` itself is untouched. Latent slots and
    latent globals keep their proposed values.
    """
    expression = express(genome, space)
    blocks = expression.genome.blocks + genome.blocks[genome.n_blocks :]
    return replace(genome, blocks=blocks)


def _repair_scope(index: int, block: BlockGenes, space: SearchSpace, log: list[Repair]) -> BlockGenes:
    if block.precision not in space.precision_values(block.op):
        fallback = PRECISION_FALLBACK[block.precision]
        log.append(Repair(RULE_CONDITIONAL_SCOPE, f"blocks[{index}].precision", block.precision, fallback))
        block = replace(block, precision=fallback)
    if block.sparsity is not None and block.sparsity not in space.sparsity_values(block.op):
        log.append(Repair(RULE_CONDITIONAL_SCOPE, f"blocks[{index}].sparsity", block.sparsity, "dense"))
        block = replace(block, sparsity="dense")
    return block


def _repair_reductions(active: list[BlockGenes], log: list[Repair]) -> list[BlockGenes]:
    reducing = [i for i, b in enumerate(active) if b.downsample != "none"]
    out = list(active)
    for i in reversed(reducing[MAX_SPATIAL_REDUCTIONS:]):
        log.append(Repair(RULE_MAX_REDUCTIONS, f"blocks[{i}].downsample", out[i].downsample, "none"))
        out[i] = replace(out[i], downsample="none")
    return out


# --- derived topology --------------------------------------------------------

Shortcut = Literal["none", "identity", "projection"]


def pruned_width(channels: int, prune_pct: int) -> int:
    """Channels kept after structured pruning (TCC-02 rule 3).

    ``channels * (1 - prune_pct/100)`` rounded half-up to a multiple of 16,
    never below 16 nor above ``channels``. Integer arithmetic only.
    """
    if channels <= 0 or channels % CHANNEL_ROUNDING or not 0 <= prune_pct < 100:
        raise ValueError(f"cannot prune {channels} channels by {prune_pct}%")
    numerator = channels * (100 - prune_pct)
    unit = 100 * CHANNEL_ROUNDING
    units = (2 * numerator + unit) // (2 * unit)
    return min(channels, max(CHANNEL_ROUNDING, units * CHANNEL_ROUNDING))


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """Shapes implied by one expressed block, before any residual coupling of pruning."""

    index: int
    op: str
    in_channels: int
    out_channels: int
    in_resolution: int
    out_resolution: int
    shortcut: Shortcut
    groups: int
    hidden_channels: int | None
    requested_prune_pct: int
    nominal_kept_channels: int

    @property
    def nominal_realized_prune(self) -> float:
        return 1 - self.nominal_kept_channels / self.out_channels


@dataclass(frozen=True, slots=True)
class NetworkPlan:
    stem_channels: int
    blocks: tuple[BlockPlan, ...]
    head_in_channels: int
    head_hidden_units: int | None
    num_classes: int

    @property
    def final_resolution(self) -> int:
        return self.blocks[-1].out_resolution

    @property
    def identity_coupled_blocks(self) -> tuple[int, ...]:
        """Blocks whose pruning also touches their input tensor through an identity add.

        The joint ratio for such groups is an open decision; ``nominal_kept_channels`` ignores it.
        """
        return tuple(b.index for b in self.blocks if b.shortcut == "identity")


def plan_network(expressed: ExpressedGenome) -> NetworkPlan:
    """Channel/resolution/shortcut plan used by the model builder and by invariants."""
    plans: list[BlockPlan] = []
    in_channels = expressed.stem_channels
    resolution = INPUT_RESOLUTION
    for index, block in enumerate(expressed.blocks):
        out_resolution = resolution // 2 if block.downsample != "none" else resolution
        if not block.residual:
            shortcut: Shortcut = "none"
        elif in_channels == block.channels and out_resolution == resolution:
            shortcut = "identity"
        else:
            shortcut = "projection"
        if block.op == "grouped_g4":
            groups = GROUPED_CONV_GROUPS
        elif block.op == "dw_separable":
            groups = in_channels
        else:
            groups = 1
        expansion = MBCONV_EXPANSION.get(block.op)
        plans.append(
            BlockPlan(
                index=index,
                op=block.op,
                in_channels=in_channels,
                out_channels=block.channels,
                in_resolution=resolution,
                out_resolution=out_resolution,
                shortcut=shortcut,
                groups=groups,
                hidden_channels=in_channels * expansion if expansion else None,
                requested_prune_pct=block.prune_pct,
                nominal_kept_channels=pruned_width(block.channels, block.prune_pct),
            )
        )
        in_channels = block.channels
        resolution = out_resolution
    return NetworkPlan(
        stem_channels=expressed.stem_channels,
        blocks=tuple(plans),
        head_in_channels=plans[-1].nominal_kept_channels,
        head_hidden_units=HEAD_HIDDEN_UNITS if expressed.head == "gap_fc256_linear" else None,
        num_classes=NUM_CLASSES,
    )


def _topology_violation(plan: NetworkPlan) -> str | None:
    for b in plan.blocks:
        if b.op == "grouped_g4":
            widths = (b.in_channels, b.out_channels, b.nominal_kept_channels)
            if any(w % GROUPED_CONV_GROUPS for w in widths):
                return f"R2_grouped_divisibility: block {b.index} widths {widths} not divisible by 4"
        if b.out_resolution < INPUT_RESOLUTION >> MAX_SPATIAL_REDUCTIONS:
            return f"R1_resolution: block {b.index} resolution {b.out_resolution} below 2"
    return None
