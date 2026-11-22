"""Initial sampling: uniform per gene over its *conditionally valid* values (TCC-02 section 4).

Randomness comes only from :meth:`random.Random.random`, the single stream
CPython promises to keep stable across versions for an integer seed; choices
and intervals are derived from it here, so a stored seed/state reproduces the
same genomes after an interpreter upgrade.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from cnnlab.tcc.science.genome import BlockGenes, Genome
from cnnlab.tcc.science.space import (
    MAX_SPATIAL_REDUCTIONS,
    N_BLOCK_SLOTS,
    Choice,
    Interval,
    SearchSpace,
)

T = TypeVar("T")
RNG_ALGORITHM = "cpython-mt19937/random-only"


class GeneRng:
    """Seeded generator whose serialized state fits in a ``SearchCheckpoint``."""

    __slots__ = ("_random",)

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative int")
        self._random = random.Random(seed)

    def unit(self) -> float:
        """Uniform in [0, 1)."""
        return self._random.random()

    def index(self, n: int) -> int:
        if n <= 0:
            raise ValueError("cannot choose from an empty domain")
        return min(int(self.unit() * n), n - 1)

    def choice(self, values: Sequence[T]) -> T:
        return values[self.index(len(values))]

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.unit()

    def get_state(self) -> dict[str, Any]:
        version, internal, gauss_next = self._random.getstate()
        return {
            "algorithm": RNG_ALGORITHM,
            "version": version,
            "internal": list(internal),
            "gauss_next": gauss_next,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> GeneRng:
        if state.get("algorithm") != RNG_ALGORITHM:
            raise ValueError(f"unsupported RNG state {state.get('algorithm')!r}")
        rng = cls(0)
        rng._random.setstate((state["version"], tuple(state["internal"]), state["gauss_next"]))
        return rng


def sample_genome(space: SearchSpace, rng: GeneRng) -> Genome:
    """Draw one raw genome; fresh samples never need repair.

    Draw order is part of the reproducibility contract: global genes in the
    order of ``space.global_domains()``, then slots 0..5 in the order of
    ``space.block_domains()``. Exactly one draw per gene, even when the
    conditionally valid set has a single value, so the stream stays aligned.
    """
    globals_: dict[str, Any] = {}
    for name, domain in space.global_domains():
        globals_[name] = _draw(domain, rng)
    n_blocks = globals_["n_blocks"]

    blocks: list[BlockGenes] = []
    reductions = 0
    for slot in range(N_BLOCK_SLOTS):
        active = slot < n_blocks
        genes: dict[str, Any] = {}
        for name, domain in space.block_domains():
            if name == "downsample" and active and reductions >= MAX_SPATIAL_REDUCTIONS:
                genes[name] = rng.choice(("none",))
            elif name == "precision":
                genes[name] = rng.choice(space.precision_values(genes["op"]))
            elif name == "sparsity":
                genes[name] = rng.choice(space.sparsity_values(genes["op"]))
            else:
                genes[name] = _draw(domain, rng)
        if active and genes["downsample"] != "none":
            reductions += 1
        blocks.append(BlockGenes(**genes))
    return Genome(**globals_, blocks=tuple(blocks))


def _draw(domain: Choice | Interval, rng: GeneRng) -> Any:
    if isinstance(domain, Interval):
        return rng.uniform(domain.low, domain.high)
    return rng.choice(domain.values)
