"""Genetic operators on the raw genome and the FP32 search slice."""

from __future__ import annotations

import dataclasses

import pytest

from cnnlab.tcc.science.genome import Genome, validate_genome
from cnnlab.tcc.science.operators import (
    FP32_SLICE,
    crossover,
    make_offspring,
    mutate,
    mutate_value,
    prepare_in_slice,
    slice_frozen_genes,
)
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1, Choice, Interval

SPACE_HASH = SPACE_V1.space_hash()
SLICE_HASH = FP32_SLICE.space_hash()


def context(*, space_hash: str = SLICE_HASH) -> CacheContext:
    return CacheContext(
        split_fingerprint="split-45k-5k-cal1024-v1",
        train_seed=101,
        calibration_seed=202,
        train_policy={"epochs": 5, "batch": 128, "optimizer": "adamw", "schedule": "cosine+1warmup"},
        pruning_policy={"fine_tune_epochs": 1},
        quantization_policy={"qat_epochs": 1, "qdq": "explicit-symmetric"},
        code_version="b3612d3",
        backend={"torch": "2.13.0+cu130", "tensorrt": None},
        builder_config={"strongly_typed": True},
        space_hash=space_hash,
    )


def genomes(seed: int, n: int) -> list[Genome]:
    rng = GeneRng(seed)
    return [sample_genome(SPACE_V1, rng) for _ in range(n)]


def slice_genomes(seed: int, n: int) -> list[Genome]:
    rng = GeneRng(seed)
    return [FP32_SLICE.sample(rng) for _ in range(n)]


# --- SearchSlice ---------------------------------------------------------------


def test_fp32_slice_pin_forces_precisions_and_prune_pct() -> None:
    g = genomes(1, 1)[0]
    pinned = FP32_SLICE.pin(g)
    assert pinned.stem_precision == "fp32" and pinned.head_precision == "fp32"
    assert all(b.precision == "fp32" and b.prune_pct == 0 for b in pinned.blocks)


def test_fp32_slice_contains_true_only_after_pinning() -> None:
    g = genomes(2, 1)[0]
    pinned = FP32_SLICE.pin(g)
    assert FP32_SLICE.contains(pinned)
    # Extremely unlikely that a fresh sample already lands in the slice.
    unpinned = [g2 for g2 in genomes(3, 50) if not FP32_SLICE.contains(g2)]
    assert unpinned, "expected at least one raw sample outside the FP32 slice"


def test_fp32_slice_sample_stays_in_slice() -> None:
    for g in slice_genomes(4, 30):
        assert FP32_SLICE.contains(g)
        validate_genome(g, SPACE_V1)


def test_slice_hash_differs_from_base_space_hash() -> None:
    assert SLICE_HASH != SPACE_HASH


def test_slice_frozen_genes_names_globals_and_block_prefixed() -> None:
    frozen = slice_frozen_genes(FP32_SLICE)
    assert frozen == frozenset({"stem_precision", "head_precision", "block.precision", "block.prune_pct"})


# --- prepare_in_slice ------------------------------------------------------------


def test_prepare_in_slice_rejects_genome_outside_slice() -> None:
    g = genomes(5, 1)[0]
    assert not FP32_SLICE.contains(g)
    with pytest.raises(ValueError):
        prepare_in_slice(g, FP32_SLICE, context())


def test_prepare_in_slice_rejects_context_for_another_space() -> None:
    g = slice_genomes(6, 1)[0]
    with pytest.raises(ValueError):
        prepare_in_slice(g, FP32_SLICE, context(space_hash=SPACE_HASH))


def test_prepare_in_slice_produces_a_slice_hashed_record() -> None:
    g = slice_genomes(7, 1)[0]
    prepared = prepare_in_slice(g, FP32_SLICE, context())
    assert prepared.record.space_hash == SLICE_HASH
    assert prepared.record.raw_genome == g.to_dict()


# --- crossover -------------------------------------------------------------------


def test_crossover_children_are_complementary_and_in_domain() -> None:
    rng = GeneRng(10)
    for _ in range(30):
        a, b = genomes(rng.index(10_000), 1)[0], genomes(rng.index(10_000) + 1, 1)[0]
        c1, c2 = crossover(a, b, rng, SPACE_V1)
        validate_genome(c1, SPACE_V1)
        validate_genome(c2, SPACE_V1)
        for name, _ in SPACE_V1.global_domains():
            v1, v2 = getattr(c1, name), getattr(c2, name)
            va, vb = getattr(a, name), getattr(b, name)
            assert {v1, v2} == {va, vb}
        for slot in range(len(a.blocks)):
            pair1, pair2 = c1.blocks[slot], c2.blocks[slot]
            assert (pair1, pair2) in ((a.blocks[slot], b.blocks[slot]), (b.blocks[slot], a.blocks[slot]))


def test_crossover_is_reproducible_for_the_same_rng_seed() -> None:
    a, b = genomes(20, 2)
    c1a, c2a = crossover(a, b, GeneRng(99), SPACE_V1)
    c1b, c2b = crossover(a, b, GeneRng(99), SPACE_V1)
    assert (c1a, c2a) == (c1b, c2b)


# --- mutate_value ----------------------------------------------------------------


def test_mutate_value_ordinal_moves_exactly_one_step() -> None:
    domain = Choice((16, 32, 64), "ordinal")
    rng = GeneRng(1)
    assert mutate_value(domain, 32, rng, 0.15) in (16, 64)
    assert mutate_value(domain, 16, rng, 0.15) == 32  # clamped at the low edge
    assert mutate_value(domain, 64, rng, 0.15) == 32  # clamped at the high edge


def test_mutate_value_categorical_never_returns_the_same_value() -> None:
    domain = Choice(("relu", "silu", "hardswish"), "categorical")
    rng = GeneRng(2)
    for _ in range(50):
        assert mutate_value(domain, "relu", rng, 0.15) != "relu"


def test_mutate_value_categorical_of_size_one_is_a_noop() -> None:
    domain = Choice(("only",), "categorical")
    assert mutate_value(domain, "only", GeneRng(3), 0.15) == "only"


def test_mutate_value_continuous_stays_within_bounds() -> None:
    domain = Interval(-4.0, -2.0, "log10_exponent")
    rng = GeneRng(4)
    for _ in range(500):
        v = mutate_value(domain, -3.0, rng, 0.5)
        assert domain.low <= v <= domain.high
        assert isinstance(v, float)


def test_mutate_value_type_matching_distinguishes_bool_from_int() -> None:
    domain = Choice((False, True), "categorical")
    rng = GeneRng(5)
    assert mutate_value(domain, False, rng, 0.15) is True
    assert mutate_value(domain, True, rng, 0.15) is False


# --- mutate ------------------------------------------------------------------------


def test_mutate_always_changes_at_least_one_gene() -> None:
    for seed in range(30):
        g = genomes(seed, 1)[0]
        child = mutate(g, GeneRng(seed + 1), SPACE_V1)
        assert child != g
        validate_genome(child, SPACE_V1)


def test_mutate_respects_frozen_genes() -> None:
    frozen = frozenset({"stem_precision", "head_precision", "block.precision", "block.prune_pct"})
    for seed in range(30):
        g = slice_genomes(seed, 1)[0]
        child = mutate(g, GeneRng(seed + 100), SPACE_V1, frozen=frozen)
        assert child.stem_precision == g.stem_precision == "fp32"
        assert child.head_precision == g.head_precision == "fp32"
        for cb, gb in zip(child.blocks, g.blocks):
            assert cb.precision == gb.precision == "fp32"
            assert cb.prune_pct == gb.prune_pct == 0
        # Freezing every pinned gene still leaves room to mutate something else.
        assert child != g


def test_mutate_ordinal_gene_moves_one_step_not_further() -> None:
    # n_blocks is an ordinal global gene with domain (2, 3, 4, 5, 6).
    g = dataclasses.replace(genomes(1, 1)[0], n_blocks=4)
    validate_genome(g, SPACE_V1)
    values = set()
    for seed in range(200):
        child = mutate(g, GeneRng(seed), SPACE_V1)
        if child.n_blocks != g.n_blocks:
            values.add(child.n_blocks)
    assert values <= {3, 5}


def test_mutate_continuous_gene_stays_in_domain_over_many_seeds() -> None:
    for seed in range(200):
        g = genomes(seed, 1)[0]
        child = mutate(g, GeneRng(seed + 1), SPACE_V1, sigma_rel=0.5)
        assert -4.0 <= child.log10_lr <= -2.0
        assert -6.0 <= child.log10_wd <= -3.0
        assert 0.0 <= child.head_dropout <= 0.5


# --- make_offspring --------------------------------------------------------------


def test_make_offspring_is_deterministic_for_the_same_seed() -> None:
    a, b = slice_genomes(30, 2)
    r1 = make_offspring(a, b, GeneRng(555), FP32_SLICE)
    r2 = make_offspring(a, b, GeneRng(555), FP32_SLICE)
    assert r1 == r2


def test_make_offspring_never_returns_an_unchanged_parent() -> None:
    for seed in range(50):
        a, b = slice_genomes(seed, 2)
        if a == b:
            continue
        c1, c2, info = make_offspring(a, b, GeneRng(seed + 1000), FP32_SLICE)
        assert c1 != a and c1 != b
        assert c2 != a and c2 != b
        assert isinstance(info["crossover"], bool)
        assert isinstance(info["mutated"], list) and len(info["mutated"]) == 2


def test_make_offspring_children_stay_inside_the_slice() -> None:
    for seed in range(50):
        a, b = slice_genomes(seed, 2)
        c1, c2, _ = make_offspring(a, b, GeneRng(seed + 2000), FP32_SLICE)
        assert FP32_SLICE.contains(c1)
        assert FP32_SLICE.contains(c2)
        validate_genome(c1, SPACE_V1)
        validate_genome(c2, SPACE_V1)
