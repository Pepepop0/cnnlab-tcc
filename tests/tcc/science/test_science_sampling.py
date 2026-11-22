"""Initial sampling: determinism, resumable RNG state, validity and domain coverage."""

from __future__ import annotations

from collections import Counter

import pytest

from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import validate_genome
from cnnlab.tcc.science.hashing import canonical_json, digest
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import MAX_SPATIAL_REDUCTIONS, SPACE_V1, Choice, Interval

N = 3000

# Guards the draw order and the RNG derivation: a change here silently alters
# which genomes a stored seed reproduces.
SEED_0_FIRST_GENOME = "e2ae659628e6420fad21608c03f5d9957d2339d5bdfbbacf855ecee768fa7220"


def samples(seed: int, n: int = N) -> list:
    rng = GeneRng(seed)
    return [sample_genome(SPACE_V1, rng) for _ in range(n)]


def test_same_seed_same_genomes() -> None:
    assert samples(123, 50) == samples(123, 50)


def test_different_seeds_differ() -> None:
    assert samples(1, 20) != samples(2, 20)


def test_rng_state_resumes_the_same_stream() -> None:
    rng = GeneRng(99)
    for _ in range(7):
        sample_genome(SPACE_V1, rng)
    state = rng.get_state()
    canonical_json(state)  # fits a JSON checkpoint
    expected = [sample_genome(SPACE_V1, rng) for _ in range(5)]
    resumed = GeneRng.from_state(state)
    assert [sample_genome(SPACE_V1, resumed) for _ in range(5)] == expected


def test_rng_rejects_bad_seed_and_foreign_state() -> None:
    with pytest.raises(ValueError):
        GeneRng(-1)
    with pytest.raises(ValueError):
        GeneRng(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GeneRng.from_state({"algorithm": "numpy-pcg64"})


def test_first_genome_of_seed_zero_is_pinned() -> None:
    first = sample_genome(SPACE_V1, GeneRng(0))
    assert digest("test.raw-genome", first.to_dict()) == SEED_0_FIRST_GENOME


def test_fresh_samples_are_valid_and_need_no_repair() -> None:
    for g in samples(5):
        validate_genome(g, SPACE_V1)
        expression = express(g, SPACE_V1)
        assert expression.repairs == ()
        assert expression.valid
        reductions = sum(b.downsample != "none" for b in g.active_blocks)
        assert reductions <= MAX_SPATIAL_REDUCTIONS


def test_every_domain_value_is_reachable() -> None:
    drawn = samples(8)
    for name, domain in SPACE_V1.global_domains():
        values = [getattr(g, name) for g in drawn]
        if isinstance(domain, Choice):
            assert set(values) == set(domain.values), name
        else:
            assert isinstance(domain, Interval)
            assert all(domain.contains(v) for v in values)
            span = domain.high - domain.low
            assert min(values) < domain.low + 0.01 * span, name
            assert max(values) > domain.high - 0.01 * span, name
    for name, domain in SPACE_V1.block_domains():
        assert isinstance(domain, Choice)
        for slot in range(6):
            seen = {getattr(g.blocks[slot], name) for g in drawn}
            assert seen == set(domain.values), (name, slot)


def test_categorical_draws_are_roughly_uniform() -> None:
    drawn = samples(21)
    for name in ("n_blocks", "activation", "stem_precision"):
        counts = Counter(getattr(g, name) for g in drawn)
        expected = N / len(counts)
        assert all(abs(c - expected) < 0.2 * expected for c in counts.values()), (name, counts)


def test_reduction_limit_only_constrains_active_blocks() -> None:
    drawn = samples(3)
    latent_reductions = [b.downsample != "none" for g in drawn for b in g.blocks[g.n_blocks :]]
    # Latent slots keep unconstrained draws for later mutation/crossover.
    assert 0.6 < sum(latent_reductions) / len(latent_reductions) < 0.9
    six_active = [g for g in drawn if g.n_blocks == 6]
    assert any(sum(b.downsample != "none" for b in g.blocks) == 4 for g in six_active)
