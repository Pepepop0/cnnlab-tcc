"""Search slices with restricted genes: FP16 per block, and FP32 unchanged (CPU, no evaluator)."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import Genome, validate_expressed, validate_genome
from cnnlab.tcc.science.operators import (
    FP16_BLOCK_SLICE,
    FP32_SLICE,
    SLICES,
    SearchSlice,
    crossover,
    make_offspring,
    mutate,
    prepare_in_slice,
    slice_frozen_genes,
)
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

# Fixed on purpose: results stored under this hash were produced by the FP32 slice.
FP32_SLICE_HASH = "b6c32232b65759e8b376471b3869cb5c7af53700124b1f717ef8f6dffaab02aa"
FP16_BLOCK_SLICE_HASH = "fd564da8476af9fc304f9ced3d7f56c96ccd7217f3d5886d6020d024cab9ad28"
N = 300


def context(space: SearchSlice) -> CacheContext:
    return CacheContext(
        split_fingerprint="split",
        train_seed=1,
        calibration_seed=0,
        train_policy={},
        pruning_policy={},
        quantization_policy={},
        code_version="test",
        backend={},
        builder_config={},
        space_hash=space.space_hash(),
    )


def samples(space: SearchSlice, seed: int = 0, n: int = N) -> list[Genome]:
    rng = GeneRng(seed)
    return [space.sample(rng) for _ in range(n)]


# --- identity ------------------------------------------------------------------------


def test_hashes_are_fixed_and_distinct() -> None:
    assert FP32_SLICE.space_hash() == FP32_SLICE_HASH
    assert FP16_BLOCK_SLICE.space_hash() == FP16_BLOCK_SLICE_HASH
    assert len({FP32_SLICE_HASH, FP16_BLOCK_SLICE_HASH, SPACE_V1.space_hash()}) == 3
    assert set(SLICES) == {
        "fp32",
        "fp16_block",
        "fp16_regions",
        "bf16_regions",
        "int8_ptq",
        "int8_qat",
        "prune_fp32",
        "compressed",
        "admitted",
        "architecture",
    }  # regions: see test_science_slices_regions
    assert SLICES["fp32"] is FP32_SLICE and SLICES["fp16_block"] is FP16_BLOCK_SLICE


def test_fp32_slice_definition_has_no_restriction_keys() -> None:
    definition = FP32_SLICE.definition()
    assert set(definition) == {"schema", "base", "pinned_global", "pinned_block", "reason"}
    assert FP16_BLOCK_SLICE.definition()["allowed_block"] == {"precision": ["fp32", "fp16"]}
    assert dict(FP16_BLOCK_SLICE.pinned_block) == {"prune_pct": 0}
    assert dict(FP16_BLOCK_SLICE.pinned_global) == {"head_precision": "fp32", "stem_precision": "fp32"}


def test_fp32_sampling_is_unchanged_by_the_extension() -> None:
    """The FP32 slice draws exactly what it drew before restrictions existed (no extra draws)."""
    from cnnlab.tcc.science.sampling import sample_genome

    a, b = GeneRng(7), GeneRng(7)
    for _ in range(20):
        assert FP32_SLICE.sample(a) == FP32_SLICE.pin(sample_genome(SPACE_V1, b))
    assert a.get_state() == b.get_state()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_block": (("precision", ("fp16", "fp32")),)},  # not in the order of the base domain
        {"allowed_block": (("precision", ("fp32",)),)},  # a single value is a pin
        {"allowed_block": (("precision", ("fp32", "fp8")),)},  # outside the base domain
        {"allowed_block": (("precision", ("fp32", "fp16")),), "pinned_block": (("precision", "fp32"),)},
        {"allowed_block": (("precision", ("fp32", "fp16")), ("precision", ("fp32", "int8")))},
        {"allowed_global": (("log10_lr", (-3.0, -2.0)),)},  # continuous genes cannot be restricted
        {"allowed_block": (("no_such_gene", (1, 2)),)},
        {"pinned_block": (("prune_pct", 10),)},
    ],
)
def test_malformed_slices_are_refused(kwargs: dict[str, object]) -> None:
    base = {"base": SPACE_V1, "pinned_global": (), "pinned_block": (), "reason": "test"}
    with pytest.raises(ValueError):
        SearchSlice(**{**base, **kwargs})  # type: ignore[arg-type]


# --- sampling ------------------------------------------------------------------------------


def test_samples_stay_in_the_slice_and_cover_both_precisions() -> None:
    seen: Counter[str] = Counter()
    for genome in samples(FP16_BLOCK_SLICE):
        validate_genome(genome, SPACE_V1)
        assert FP16_BLOCK_SLICE.contains(genome)
        assert genome.stem_precision == genome.head_precision == "fp32"
        assert all(b.prune_pct == 0 for b in genome.blocks)
        seen.update(b.precision for b in genome.blocks)
    assert set(seen) == {"fp32", "fp16"}
    assert 0.4 < seen["fp16"] / sum(seen.values()) < 0.6


def test_sampling_is_deterministic_and_resumable() -> None:
    assert samples(FP16_BLOCK_SLICE, seed=5, n=10) == samples(FP16_BLOCK_SLICE, seed=5, n=10)
    rng = GeneRng(5)
    first = [FP16_BLOCK_SLICE.sample(rng) for _ in range(4)]
    resumed = GeneRng.from_state(rng.get_state())
    assert [FP16_BLOCK_SLICE.sample(rng) for _ in range(3)] == [FP16_BLOCK_SLICE.sample(resumed) for _ in range(3)]
    assert first == samples(FP16_BLOCK_SLICE, seed=5, n=4)


def test_contains_checks_latent_slots_too() -> None:
    genome = samples(FP16_BLOCK_SLICE, n=1)[0]
    for precision in ("bf16", "int8"):
        blocks = (*genome.blocks[:5], dataclasses.replace(genome.blocks[5], precision=precision))
        assert not FP16_BLOCK_SLICE.contains(dataclasses.replace(genome, blocks=blocks, n_blocks=2))
    assert not FP16_BLOCK_SLICE.contains(dataclasses.replace(genome, stem_precision="fp16"))
    pruned = (dataclasses.replace(genome.blocks[0], prune_pct=25), *genome.blocks[1:])
    assert not FP16_BLOCK_SLICE.contains(dataclasses.replace(genome, blocks=pruned))


def test_every_fp32_genome_belongs_to_the_fp16_block_slice() -> None:
    for genome in samples(FP32_SLICE, n=50):
        assert FP16_BLOCK_SLICE.contains(genome)
    mixed = [g for g in samples(FP16_BLOCK_SLICE, n=50) if any(b.precision == "fp16" for b in g.blocks)]
    assert mixed and not any(FP32_SLICE.contains(g) for g in mixed)


# --- operators --------------------------------------------------------------------------------


def test_mutation_keeps_precision_inside_the_subset_and_can_change_it() -> None:
    frozen = slice_frozen_genes(FP16_BLOCK_SLICE)
    restricted = FP16_BLOCK_SLICE.restricted_domains()
    assert frozen == frozenset({"head_precision", "stem_precision", "block.prune_pct"})
    assert set(restricted) == {"block.precision"} and restricted["block.precision"].values == ("fp32", "fp16")
    changed = 0
    for seed, genome in enumerate(samples(FP16_BLOCK_SLICE, seed=11)):
        child = mutate(genome, GeneRng(seed), SPACE_V1, frozen=frozen, restricted=restricted)
        assert child != genome and FP16_BLOCK_SLICE.contains(child)
        changed += [b.precision for b in child.active_blocks] != [b.precision for b in genome.active_blocks]
    assert changed > 0


def test_unrestricted_mutation_would_leave_the_slice() -> None:
    """The restriction matters: mutating in the base domain reaches BF16/INT8."""
    frozen = slice_frozen_genes(FP16_BLOCK_SLICE)
    outside = 0
    for seed, genome in enumerate(samples(FP16_BLOCK_SLICE, seed=11)):
        outside += not FP16_BLOCK_SLICE.contains(mutate(genome, GeneRng(seed), SPACE_V1, frozen=frozen))
    assert outside > 0


def test_crossover_and_offspring_stay_in_the_slice() -> None:
    parents = samples(FP16_BLOCK_SLICE, seed=3, n=120)
    for seed, (a, b) in enumerate(zip(parents[0::2], parents[1::2])):
        x, y = crossover(a, b, GeneRng(seed), SPACE_V1)
        assert FP16_BLOCK_SLICE.contains(x) and FP16_BLOCK_SLICE.contains(y)
        c1, c2, info = make_offspring(a, b, GeneRng(seed + 500), FP16_BLOCK_SLICE)
        assert FP16_BLOCK_SLICE.contains(c1) and FP16_BLOCK_SLICE.contains(c2)
        assert c1 not in (a, b) and c2 not in (a, b)
        assert (c1, c2, info) == make_offspring(a, b, GeneRng(seed + 500), FP16_BLOCK_SLICE)


# --- expression and hashes ----------------------------------------------------------------------


def test_expression_keeps_block_precisions_and_needs_no_int8_genes() -> None:
    for genome in samples(FP16_BLOCK_SLICE, seed=2, n=100):
        expression = express(genome, SPACE_V1)
        assert expression.valid
        validate_expressed(expression.genome, SPACE_V1)
        expressed = expression.genome
        assert [b.precision for b in expressed.blocks] == [b.precision for b in genome.active_blocks]
        assert expressed.int8_mode is None and expressed.weight_granularity is None and expressed.calibration is None
        assert not [r for r in expression.repairs if "precision" in r.path]


def test_block_precision_changes_the_phenotype_and_latent_precision_does_not() -> None:
    genome = dataclasses.replace(samples(FP16_BLOCK_SLICE, seed=9, n=1)[0], n_blocks=3)
    ctx = context(FP16_BLOCK_SLICE)
    base = prepare_in_slice(genome, FP16_BLOCK_SLICE, ctx).record

    def flipped(slot: int) -> Genome:
        block = genome.blocks[slot]
        other = "fp16" if block.precision == "fp32" else "fp32"
        blocks = (*genome.blocks[:slot], dataclasses.replace(block, precision=other), *genome.blocks[slot + 1 :])
        return dataclasses.replace(genome, blocks=blocks)

    active = prepare_in_slice(flipped(1), FP16_BLOCK_SLICE, ctx).record
    latent = prepare_in_slice(flipped(4), FP16_BLOCK_SLICE, ctx).record
    assert active.phenotype_hash != base.phenotype_hash
    assert active.cache_context_hash != base.cache_context_hash
    assert latent.phenotype_hash == base.phenotype_hash and latent.cache_context_hash == base.cache_context_hash
    assert base.space_hash == FP16_BLOCK_SLICE_HASH


def test_same_genome_has_another_identity_in_another_slice() -> None:
    """An all-FP32 genome is one phenotype, but its cached result belongs to the slice that produced it."""
    genome = samples(FP32_SLICE, seed=1, n=1)[0]
    in_fp32 = prepare_in_slice(genome, FP32_SLICE, context(FP32_SLICE)).record
    in_fp16 = prepare_in_slice(genome, FP16_BLOCK_SLICE, context(FP16_BLOCK_SLICE)).record
    assert in_fp32.phenotype_hash == in_fp16.phenotype_hash
    assert in_fp32.space_hash != in_fp16.space_hash
    assert in_fp32.cache_context_hash != in_fp16.cache_context_hash
    with pytest.raises(ValueError, match="another search space"):
        prepare_in_slice(genome, FP16_BLOCK_SLICE, context(FP32_SLICE))


def test_genome_outside_the_slice_is_refused() -> None:
    genome = samples(FP16_BLOCK_SLICE, n=1)[0]
    blocks = (dataclasses.replace(genome.blocks[0], precision="bf16"), *genome.blocks[1:])
    with pytest.raises(ValueError, match="outside the slice"):
        prepare_in_slice(dataclasses.replace(genome, blocks=blocks), FP16_BLOCK_SLICE, context(FP16_BLOCK_SLICE))
