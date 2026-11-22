"""FP16_REGIONS_SLICE: stem, blocks and head choose between fp32 and fp16 (CPU, no evaluator)."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import validate_expressed, validate_genome
from cnnlab.tcc.science.operators import (
    FP16_BLOCK_SLICE,
    FP16_REGIONS_SLICE,
    FP32_SLICE,
    PROBE_SLICES,
    SLICES,
    crossover,
    make_offspring,
    mutate,
    prepare_in_slice,
    slice_frozen_genes,
)
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

FP32_SLICE_HASH = "b6c32232b65759e8b376471b3869cb5c7af53700124b1f717ef8f6dffaab02aa"
FP16_BLOCK_SLICE_HASH = "fd564da8476af9fc304f9ced3d7f56c96ccd7217f3d5886d6020d024cab9ad28"
FP16_REGIONS_SLICE_HASH = "be60a87c10eb3369812747684db2246b3c8bd3f27b9561238e120208911eea93"
BF16_REGIONS_SLICE_HASH = "5ee08c29152927d5375ac1a8e61771e61917f7a9de2d5c9c45312d3fb43a67ae"
INT8_PTQ_SLICE_HASH = "0a509646fe79b3cbf0e384a76a0cf60ce43b2d0faa5c5ce317ce4660554ad3e7"
N = 300


def context(space: object) -> CacheContext:
    return CacheContext(
        split_fingerprint="s",
        train_seed=1,
        calibration_seed=0,
        train_policy={},
        pruning_policy={},
        quantization_policy={},
        code_version="t",
        backend={},
        builder_config={},
        space_hash=space.space_hash(),
    )  # type: ignore[attr-defined]


def samples(seed: int = 0, n: int = N) -> list:
    rng = GeneRng(seed)
    return [FP16_REGIONS_SLICE.sample(rng) for _ in range(n)]


def test_hashes_are_fixed_and_the_earlier_slices_are_unchanged() -> None:
    assert FP16_REGIONS_SLICE.space_hash() == FP16_REGIONS_SLICE_HASH
    assert FP32_SLICE.space_hash() == FP32_SLICE_HASH and FP16_BLOCK_SLICE.space_hash() == FP16_BLOCK_SLICE_HASH
    assert len({FP32_SLICE_HASH, FP16_BLOCK_SLICE_HASH, FP16_REGIONS_SLICE_HASH, SPACE_V1.space_hash()}) == 4


def test_the_bf16_slice_has_its_own_fixed_hash() -> None:
    from cnnlab.tcc.science.operators import BF16_REGIONS_SLICE

    assert BF16_REGIONS_SLICE.space_hash() == BF16_REGIONS_SLICE_HASH
    assert BF16_REGIONS_SLICE_HASH not in (FP16_REGIONS_SLICE_HASH, FP32_SLICE_HASH, FP16_BLOCK_SLICE_HASH)


def test_the_int8_slice_has_its_own_fixed_hash() -> None:
    from cnnlab.tcc.science.operators import INT8_PTQ_SLICE

    assert INT8_PTQ_SLICE.space_hash() == INT8_PTQ_SLICE_HASH
    assert INT8_PTQ_SLICE_HASH not in (FP16_REGIONS_SLICE_HASH, BF16_REGIONS_SLICE_HASH, FP32_SLICE_HASH)


def test_the_slice_is_registered_for_the_wiring_and_the_probe() -> None:
    assert SLICES["fp16_regions"] is FP16_REGIONS_SLICE and PROBE_SLICES["fp16_regions"] is FP16_REGIONS_SLICE
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
    }


def test_definition_names_the_restricted_regions() -> None:
    definition = FP16_REGIONS_SLICE.definition()
    assert definition["allowed_global"] == {"stem_precision": ["fp32", "fp16"], "head_precision": ["fp32", "fp16"]}
    assert definition["allowed_block"] == {"precision": ["fp32", "fp16"]}
    assert definition["pinned_global"] == {} and definition["pinned_block"] == {"prune_pct": 0}


def test_samples_cover_stem_head_and_blocks_and_never_leave_the_slice() -> None:
    seen: dict[str, Counter] = {"stem": Counter(), "head": Counter(), "blocks": Counter()}
    for genome in samples():
        validate_genome(genome, SPACE_V1)
        assert FP16_REGIONS_SLICE.contains(genome) and all(b.prune_pct == 0 for b in genome.blocks)
        seen["stem"][genome.stem_precision] += 1
        seen["head"][genome.head_precision] += 1
        seen["blocks"].update(b.precision for b in genome.blocks)
    for counter in seen.values():
        assert set(counter) == {"fp32", "fp16"} and 0.4 < counter["fp16"] / sum(counter.values()) < 0.6


def test_every_earlier_slice_is_inside_this_one_and_not_the_other_way() -> None:
    for earlier in (FP32_SLICE, FP16_BLOCK_SLICE):
        rng = GeneRng(2)
        assert all(FP16_REGIONS_SLICE.contains(earlier.sample(rng)) for _ in range(40))
    wider = [g for g in samples(n=60) if g.stem_precision == "fp16" or g.head_precision == "fp16"]
    assert wider and not any(FP16_BLOCK_SLICE.contains(g) for g in wider)


def test_contains_refuses_bf16_int8_and_pruning_in_any_region() -> None:
    genome = samples(n=1)[0]
    for kwargs in ({"stem_precision": "bf16"}, {"head_precision": "int8"}):
        assert not FP16_REGIONS_SLICE.contains(dataclasses.replace(genome, **kwargs))
    blocks = (dataclasses.replace(genome.blocks[5], precision="bf16"), *genome.blocks[:5])
    assert not FP16_REGIONS_SLICE.contains(dataclasses.replace(genome, blocks=blocks))
    blocks = (dataclasses.replace(genome.blocks[0], prune_pct=25), *genome.blocks[1:])
    assert not FP16_REGIONS_SLICE.contains(dataclasses.replace(genome, blocks=blocks))


def test_mutation_moves_stem_and_head_inside_their_subset() -> None:
    frozen = slice_frozen_genes(FP16_REGIONS_SLICE)
    restricted = FP16_REGIONS_SLICE.restricted_domains()
    assert frozen == frozenset({"block.prune_pct"})
    assert set(restricted) == {"stem_precision", "head_precision", "block.precision"}
    changed = Counter()
    for seed, genome in enumerate(samples(seed=7)):
        child = mutate(genome, GeneRng(seed), SPACE_V1, frozen=frozen, restricted=restricted)
        assert child != genome and FP16_REGIONS_SLICE.contains(child)
        changed["stem"] += child.stem_precision != genome.stem_precision
        changed["head"] += child.head_precision != genome.head_precision
    assert changed["stem"] > 0 and changed["head"] > 0


def test_crossover_and_offspring_stay_inside() -> None:
    parents = samples(seed=3, n=120)
    for seed, (a, b) in enumerate(zip(parents[0::2], parents[1::2])):
        x, y = crossover(a, b, GeneRng(seed), SPACE_V1)
        assert FP16_REGIONS_SLICE.contains(x) and FP16_REGIONS_SLICE.contains(y)
        c1, c2, _ = make_offspring(a, b, GeneRng(seed + 500), FP16_REGIONS_SLICE)
        assert FP16_REGIONS_SLICE.contains(c1) and FP16_REGIONS_SLICE.contains(c2)
        assert c1 not in (a, b) and c2 not in (a, b)


def test_expression_keeps_stem_and_head_and_needs_no_int8_genes() -> None:
    for genome in samples(seed=2, n=100):
        expression = express(genome, SPACE_V1)
        assert expression.valid
        validate_expressed(expression.genome, SPACE_V1)
        expressed = expression.genome
        assert (expressed.stem_precision, expressed.head_precision) == (genome.stem_precision, genome.head_precision)
        assert [b.precision for b in expressed.blocks] == [b.precision for b in genome.active_blocks]
        assert expressed.int8_mode is None and expressed.weight_granularity is None and expressed.calibration is None
        assert not [r for r in expression.repairs if "precision" in r.path]


@pytest.mark.parametrize("gene", ["stem_precision", "head_precision"])
def test_stem_and_head_precision_change_the_phenotype_and_the_cache_identity(gene: str) -> None:
    genome = samples(seed=9, n=1)[0]
    ctx = context(FP16_REGIONS_SLICE)
    base = prepare_in_slice(genome, FP16_REGIONS_SLICE, ctx).record
    other = "fp16" if getattr(genome, gene) == "fp32" else "fp32"
    changed = prepare_in_slice(dataclasses.replace(genome, **{gene: other}), FP16_REGIONS_SLICE, ctx).record
    assert changed.phenotype_hash != base.phenotype_hash and changed.cache_context_hash != base.cache_context_hash
    assert base.space_hash == FP16_REGIONS_SLICE_HASH


def test_the_same_genome_has_another_cache_identity_in_another_slice() -> None:
    genome = FP32_SLICE.sample(GeneRng(1))
    a = prepare_in_slice(genome, FP32_SLICE, context(FP32_SLICE)).record
    b = prepare_in_slice(genome, FP16_REGIONS_SLICE, context(FP16_REGIONS_SLICE)).record
    assert a.phenotype_hash == b.phenotype_hash and a.cache_context_hash != b.cache_context_hash
    with pytest.raises(ValueError, match="another search space"):
        prepare_in_slice(genome, FP16_REGIONS_SLICE, context(FP32_SLICE))
