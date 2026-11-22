"""Raw and expressed genome types, strict parsing and domain validation.

``Genome`` is the fixed-length raw genome (six block slots, latent ones kept for
mutation/crossover). ``ExpressedGenome`` holds only what shapes the network:
active blocks and the quantization genes that some INT8 part actually uses.
Both are frozen, so a repair can only ever build new objects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cnnlab.tcc.science.space import (
    BLOCK_GENES,
    CONTINUOUS_GENES,
    EXPRESSED_SCHEMA,
    GENOME_SCHEMA,
    GLOBAL_GENES,
    INT8_GENES,
    MAX_SPATIAL_REDUCTIONS,
    N_BLOCK_SLOTS,
    SearchSpace,
)


class GenomeError(ValueError):
    """Base class for malformed genomes. Never a candidate-level invalidity."""


class GenomeStructureError(GenomeError):
    """Wrong keys, types or number of block slots."""


class GenomeDomainError(GenomeError):
    """A gene value lies outside the domain of the search space."""

    def __init__(self, path: str, value: object, allowed: object) -> None:
        super().__init__(f"{path}={value!r} outside domain {allowed}")
        self.path = path
        self.value = value


_INT_FIELDS = {"n_blocks", "stem_channels", "channels", "kernel", "n_convs", "prune_pct"}
_STR_FIELDS = {
    "activation",
    "head",
    "stem_precision",
    "head_precision",
    "int8_mode",
    "weight_granularity",
    "calibration",
    "op",
    "downsample",
    "precision",
}


def _check_type(name: str, value: object, path: str) -> None:
    if name in _INT_FIELDS and type(value) is not int:
        raise GenomeStructureError(f"{path} must be int, got {type(value).__name__}")
    if name in _STR_FIELDS and type(value) is not str:
        raise GenomeStructureError(f"{path} must be str, got {type(value).__name__}")
    if name == "residual" and type(value) is not bool:
        raise GenomeStructureError(f"{path} must be bool, got {type(value).__name__}")


def _as_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenomeStructureError(f"{path} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise GenomeStructureError(f"{path} must be finite")
    return 0.0 if number == 0.0 else number


@dataclass(frozen=True, slots=True)
class BlockGenes:
    """Genes of one block slot. ``sparsity`` exists only if 2:4 was admitted."""

    channels: int
    op: str
    kernel: int
    n_convs: int
    residual: bool
    downsample: str
    precision: str
    prune_pct: int
    sparsity: str | None = None

    def __post_init__(self) -> None:
        for name in BLOCK_GENES:
            _check_type(name, getattr(self, name), f"block.{name}")
        if self.sparsity is not None and type(self.sparsity) is not str:
            raise GenomeStructureError("block.sparsity must be str or None")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name) for name in BLOCK_GENES}
        if self.sparsity is not None:
            out["sparsity"] = self.sparsity
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str = "block") -> BlockGenes:
        allowed = set(BLOCK_GENES) | {"sparsity"}
        _check_keys(data, set(BLOCK_GENES), allowed, path)
        return cls(**{k: data[k] for k in data})


@dataclass(frozen=True, slots=True)
class Genome:
    """Raw genome as proposed by sampling or by a genetic operator."""

    n_blocks: int
    stem_channels: int
    activation: str
    head: str
    stem_precision: str
    head_precision: str
    int8_mode: str
    weight_granularity: str
    calibration: str
    log10_lr: float
    log10_wd: float
    head_dropout: float
    blocks: tuple[BlockGenes, ...]

    def __post_init__(self) -> None:
        for name in GLOBAL_GENES:
            if name in CONTINUOUS_GENES:
                object.__setattr__(self, name, _as_float(getattr(self, name), name))
            else:
                _check_type(name, getattr(self, name), name)
        blocks = tuple(self.blocks)
        if len(blocks) != N_BLOCK_SLOTS:
            raise GenomeStructureError(f"raw genome needs {N_BLOCK_SLOTS} block slots, got {len(blocks)}")
        if not all(isinstance(b, BlockGenes) for b in blocks):
            raise GenomeStructureError("blocks must be BlockGenes")
        object.__setattr__(self, "blocks", blocks)

    @property
    def active_blocks(self) -> tuple[BlockGenes, ...]:
        return self.blocks[: self.n_blocks]

    def to_dict(self) -> dict[str, Any]:
        """JSON-native copy; mutating it never touches this genome."""
        out: dict[str, Any] = {"schema": GENOME_SCHEMA}
        out.update({name: getattr(self, name) for name in GLOBAL_GENES})
        out["blocks"] = [b.to_dict() for b in self.blocks]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Genome:
        """Strict inverse of :meth:`to_dict` (structure only; see :func:`validate_genome`)."""
        required = {"schema", "blocks", *GLOBAL_GENES}
        _check_keys(data, required, required, "genome")
        if data["schema"] != GENOME_SCHEMA:
            raise GenomeStructureError(f"unsupported genome schema {data['schema']!r}")
        blocks = _parse_blocks(data["blocks"])
        return cls(**{name: data[name] for name in GLOBAL_GENES}, blocks=blocks)


@dataclass(frozen=True, slots=True)
class ExpressedGenome:
    """Phenotype-defining genes only. INT8 genes are ``None`` when not expressed."""

    n_blocks: int
    stem_channels: int
    activation: str
    head: str
    stem_precision: str
    head_precision: str
    int8_mode: str | None
    weight_granularity: str | None
    calibration: str | None
    log10_lr: float
    log10_wd: float
    head_dropout: float
    blocks: tuple[BlockGenes, ...]

    def __post_init__(self) -> None:
        for name in GLOBAL_GENES:
            value = getattr(self, name)
            if name in CONTINUOUS_GENES:
                object.__setattr__(self, name, _as_float(value, name))
            elif not (name in INT8_GENES and value is None):
                _check_type(name, value, name)
        blocks = tuple(self.blocks)
        if len(blocks) != self.n_blocks:
            raise GenomeStructureError("expressed genome carries exactly the active blocks")
        object.__setattr__(self, "blocks", blocks)

    @property
    def uses_int8(self) -> bool:
        return "int8" in self.precisions()

    def precisions(self) -> tuple[str, ...]:
        return (self.stem_precision, self.head_precision, *(b.precision for b in self.blocks))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": EXPRESSED_SCHEMA}
        for name in GLOBAL_GENES:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["blocks"] = [b.to_dict() for b in self.blocks]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExpressedGenome:
        optional = set(INT8_GENES)
        required = {"schema", "blocks", *GLOBAL_GENES} - optional
        _check_keys(data, required, required | optional, "expressed")
        if data["schema"] != EXPRESSED_SCHEMA:
            raise GenomeStructureError(f"unsupported expressed schema {data['schema']!r}")
        values = {name: data.get(name) for name in GLOBAL_GENES}
        blocks_raw = data["blocks"]
        if not isinstance(blocks_raw, Sequence) or isinstance(blocks_raw, (str, bytes)):
            raise GenomeStructureError("expressed.blocks must be a list")
        blocks = tuple(BlockGenes.from_dict(b, f"expressed.blocks[{i}]") for i, b in enumerate(blocks_raw))
        return cls(**values, blocks=blocks)


def _parse_blocks(raw: object) -> tuple[BlockGenes, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise GenomeStructureError("genome.blocks must be a list")
    return tuple(BlockGenes.from_dict(b, f"genome.blocks[{i}]") for i, b in enumerate(raw))


def _check_keys(data: object, required: set[str], allowed: set[str], path: str) -> None:
    if not isinstance(data, Mapping):
        raise GenomeStructureError(f"{path} must be a mapping")
    keys = set(data)
    missing = required - keys
    unknown = keys - allowed
    if missing:
        raise GenomeStructureError(f"{path} missing keys {sorted(missing)}")
    if unknown:
        raise GenomeStructureError(f"{path} has unknown keys {sorted(unknown)}")


# --- domain validation -------------------------------------------------------


def validate_genome(genome: Genome, space: SearchSpace) -> None:
    """Raise :class:`GenomeDomainError` unless every raw gene, latent included, is in the space."""
    for name, domain in space.global_domains():
        value = getattr(genome, name)
        if not domain.contains(value):
            raise GenomeDomainError(name, value, domain.definition())
    for index, block in enumerate(genome.blocks):
        _validate_block(block, space, f"blocks[{index}]")


def validate_expressed(expressed: ExpressedGenome, space: SearchSpace) -> None:
    """Check the post-repair invariants that any consumer of a phenotype may rely on."""
    for name, domain in space.global_domains():
        value = getattr(expressed, name)
        if name in INT8_GENES and value is None:
            continue
        if not domain.contains(value):
            raise GenomeDomainError(name, value, domain.definition())
    for index, block in enumerate(expressed.blocks):
        path = f"blocks[{index}]"
        _validate_block(block, space, path)
        if block.precision not in space.precision_values(block.op):
            raise GenomeDomainError(f"{path}.precision", block.precision, space.precision_values(block.op))
        if block.sparsity is not None and block.sparsity not in space.sparsity_values(block.op):
            raise GenomeDomainError(f"{path}.sparsity", block.sparsity, space.sparsity_values(block.op))
    reductions = sum(1 for b in expressed.blocks if b.downsample != "none")
    if reductions > MAX_SPATIAL_REDUCTIONS:
        raise GenomeDomainError("blocks.downsample", reductions, f"<= {MAX_SPATIAL_REDUCTIONS} active reductions")
    uses_int8 = expressed.uses_int8
    for name in INT8_GENES:
        value = getattr(expressed, name)
        expected = uses_int8 and (name != "calibration" or expressed.int8_mode == "ptq")
        if (value is not None) != expected:
            state = "expressed" if expected else "latent"
            raise GenomeDomainError(name, value, f"{state} for this phenotype")


def _validate_block(block: BlockGenes, space: SearchSpace, path: str) -> None:
    for name, domain in space.block_domains():
        value = getattr(block, name)
        if not domain.contains(value):
            raise GenomeDomainError(f"{path}.{name}", value, domain.definition())
    if not space.has_sparsity_gene and block.sparsity is not None:
        raise GenomeDomainError(f"{path}.sparsity", block.sparsity, "absent: 2:4 not admitted")
