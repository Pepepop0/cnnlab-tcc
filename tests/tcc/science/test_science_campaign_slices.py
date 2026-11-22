"""The slices of the campaign: ADMITTED_SLICE, ARCHITECTURE_SLICE and the BF16 precision scope (CPU)."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from cnnlab.tcc.model_pipeline.precision import bf16_engine_limits, representation_problems
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import validate_expressed, validate_genome
from cnnlab.tcc.science.operators import (
    ADMITTED_SLICE,
    ARCHITECTURE_SLICE,
    BF16_SCOPE,
    COMPRESSED_SLICE,
    FP32_SLICE,
    SLICES,
    PrecisionScope,
    SearchSlice,
    crossover,
    make_offspring,
    mutate,
    prepare_in_slice,
    slice_frozen_genes,
)
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.sequential import (
    ARCHITECTURE_BLOCK,
    ARCHITECTURE_GLOBAL,
    GLOBAL_GENES,
    graft,
    representation_is_searchable,
)
from cnnlab.tcc.science.space import SPACE_V1

ADMITTED_SLICE_HASH = "7466ed6506b10b1a42558f2f3b6d0ca4e1d1df8be4af6e25032bc3442b72b2a3"
ARCHITECTURE_SLICE_HASH = "b5c81dd11d57c0e6840b89b001b616a17328c7573fb7e37c5a03f751f46d5c63"
N = 400


def samples(space: SearchSlice, seed: int = 0, n: int = N) -> list:
    rng = GeneRng(seed)
    return [space.sample(rng) for _ in range(n)]


def context(space: SearchSlice) -> CacheContext:
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
    )


# --- identity ------------------------------------------------------------------------------------------------------


def test_hashes_are_fixed_distinct_and_registered() -> None:
    assert ADMITTED_SLICE.space_hash() == ADMITTED_SLICE_HASH
    assert ARCHITECTURE_SLICE.space_hash() == ARCHITECTURE_SLICE_HASH
    assert (
        len(
            {
                ADMITTED_SLICE_HASH,
                ARCHITECTURE_SLICE_HASH,
                SPACE_V1.space_hash(),
                COMPRESSED_SLICE.space_hash(),
                FP32_SLICE.space_hash(),
            }
        )
        == 5
    )
    assert SLICES["admitted"] is ADMITTED_SLICE and SLICES["architecture"] is ARCHITECTURE_SLICE


def test_both_slices_share_one_base_space() -> None:
    assert ADMITTED_SLICE.base == ARCHITECTURE_SLICE.base == SPACE_V1
    assert ADMITTED_SLICE.base.space_hash() == ARCHITECTURE_SLICE.base.space_hash()


def test_a_slice_without_scopes_keeps_the_hash_it_had() -> None:
    assert "precision_scopes" not in COMPRESSED_SLICE.definition()
    assert ADMITTED_SLICE.definition()["precision_scopes"] == [BF16_SCOPE.definition()]


def test_the_scope_and_its_evidence_are_in_the_hash_and_the_reason() -> None:
    narrower = dataclasses.replace(BF16_SCOPE, block_ops=("standard",))
    other = dataclasses.replace(ADMITTED_SLICE, scopes=(narrower,))
    assert other.space_hash() != ADMITTED_SLICE.space_hash()
    assert "TensorRT 11.3" in BF16_SCOPE.evidence and "HardSwish" in BF16_SCOPE.evidence
    for word in ("BF16", "standard and grouped_g4", "FP8", "2:4"):
        assert word in ADMITTED_SLICE.reason


def test_scope_validation_rejects_nonsense() -> None:
    base = dict(base=SPACE_V1, pinned_global=(), pinned_block=(), reason="t")
    for bad in (
        dataclasses.replace(BF16_SCOPE, fallback="bf16"),
        dataclasses.replace(BF16_SCOPE, precision="fp7"),
        dataclasses.replace(BF16_SCOPE, block_ops=("conv9",)),
        dataclasses.replace(BF16_SCOPE, stem_activations=("gelu",)),
    ):
        with pytest.raises(ValueError):
            SearchSlice(**base, scopes=(bad,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pinned to the scoped precision"):
        SearchSlice(**{**base, "pinned_global": (("stem_precision", "bf16"),)}, scopes=(BF16_SCOPE,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fallback is not allowed"):
        SearchSlice(**base, allowed_block=(("precision", ("fp32", "bf16")),), scopes=(BF16_SCOPE,))  # type: ignore[arg-type]


# --- what the scope allows -----------------------------------------------------------------------------------------


def with_regions(
    stem: str = "fp32",
    head: str = "fp32",
    block: str = "fp32",
    op: str = "standard",
    activation: str = "relu",
    head_type: str = "gap_linear",
) -> object:
    raw = ADMITTED_SLICE.sample(GeneRng(3))
    blocks = tuple(dataclasses.replace(b, op=op, precision=block) for b in raw.blocks)
    return dataclasses.replace(
        raw, activation=activation, head=head_type, stem_precision=stem, head_precision=head, blocks=blocks
    )


@pytest.mark.parametrize(
    ("kwargs", "inside"),
    [
        ({"block": "bf16", "op": "standard", "activation": "relu"}, True),
        ({"block": "bf16", "op": "grouped_g4", "activation": "silu"}, True),
        ({"block": "bf16", "op": "standard", "activation": "hardswish"}, False),
        ({"block": "bf16", "op": "dw_separable", "activation": "relu"}, False),
        ({"block": "bf16", "op": "mbconv_e3", "activation": "silu"}, False),
        ({"block": "bf16", "op": "mbconv_e6", "activation": "relu"}, False),
        ({"stem": "bf16", "activation": "silu"}, True),
        ({"stem": "bf16", "activation": "hardswish"}, False),
        ({"head": "bf16", "activation": "hardswish"}, True),  # a plain linear head has no activation to reject
        ({"head": "bf16", "activation": "hardswish", "head_type": "gap_fc256_linear"}, False),
        ({"head": "bf16", "activation": "relu", "head_type": "gap_fc256_linear"}, True),
    ],
)
def test_bf16_is_inside_the_slice_only_where_the_engine_runs_it(kwargs: dict, inside: bool) -> None:
    genome = with_regions(**kwargs)
    assert ADMITTED_SLICE.contains(genome) is inside  # type: ignore[arg-type]
    projected = ADMITTED_SLICE.pin(genome)  # type: ignore[arg-type]
    assert ADMITTED_SLICE.contains(projected) and (projected == genome) is inside


def test_pin_moves_a_forbidden_bf16_to_fp16_and_nothing_else() -> None:
    genome = with_regions(
        stem="bf16", head="bf16", block="bf16", op="mbconv_e6", activation="hardswish", head_type="gap_fc256_linear"
    )
    pinned = ADMITTED_SLICE.pin(genome)  # type: ignore[arg-type]
    assert (pinned.stem_precision, pinned.head_precision) == ("fp16", "fp16")
    assert all(b.precision == "fp16" for b in pinned.blocks)
    assert dataclasses.replace(pinned, stem_precision="bf16", head_precision="bf16", blocks=genome.blocks) == genome  # type: ignore[attr-defined]
    assert ADMITTED_SLICE.pin(pinned) == pinned  # idempotent


def test_fp16_and_int8_are_never_conditioned() -> None:
    for activation in ("relu", "silu", "hardswish"):
        for op in ("standard", "grouped_g4", "dw_separable", "mbconv_e3", "mbconv_e6"):
            for precision in ("fp32", "fp16", "int8"):
                genome = with_regions(
                    stem=precision,
                    head=precision,
                    block=precision,
                    op=op,
                    activation=activation,
                    head_type="gap_fc256_linear",
                )
                assert ADMITTED_SLICE.contains(genome)  # type: ignore[arg-type]


# --- sampling ------------------------------------------------------------------------------------------------------------


def test_samples_are_inside_the_slice_and_the_pipeline_runs_all_of_them() -> None:
    seen: Counter = Counter()
    for genome in samples(ADMITTED_SLICE, seed=1, n=1000):
        validate_genome(genome, SPACE_V1)
        assert ADMITTED_SLICE.contains(genome)
        expressed = express(genome, SPACE_V1).genome
        validate_expressed(expressed, SPACE_V1)
        assert representation_problems(expressed) == [] and bf16_engine_limits(expressed) == []
        seen.update([f"stem:{genome.stem_precision}", f"head:{genome.head_precision}"])
        seen.update(f"block:{b.precision}" for b in genome.blocks)
        seen.update(f"prune:{b.prune_pct}" for b in genome.blocks)
        seen.update(
            [
                f"mode:{genome.int8_mode}",
                f"granularity:{genome.weight_granularity}",
                f"calibration:{genome.calibration}",
            ]
        )
    for key in (
        "stem:bf16",
        "head:bf16",
        "block:bf16",
        "block:int8",
        "prune:25",
        "prune:50",
        "mode:qat",
        "mode:ptq",
        "granularity:per_tensor",
        "calibration:entropy",
    ):
        assert seen[key] > 0, key


def test_sampling_draws_uniformly_among_the_valid_values_not_from_a_projection() -> None:
    """A bad BF16 draw must not pile probability on the fallback: FP16 stays as frequent as FP32 and INT8."""
    counts: Counter = Counter()
    for genome in samples(ADMITTED_SLICE, seed=2, n=2500):
        counts.update(b.precision for b in genome.blocks)
    base = counts["fp32"]
    assert abs(counts["fp16"] - base) < 0.08 * base and abs(counts["int8"] - base) < 0.08 * base
    assert 0.1 * base < counts["bf16"] < 0.5 * base  # only where it is valid (about a quarter of the pairs)


def test_sampling_is_deterministic_and_reaches_every_valid_bf16_pair() -> None:
    assert samples(ADMITTED_SLICE, seed=5, n=30) == samples(ADMITTED_SLICE, seed=5, n=30)
    pairs = set()
    for genome in samples(ADMITTED_SLICE, seed=6, n=1500):
        pairs |= {(b.op, genome.activation) for b in genome.blocks if b.precision == "bf16"}
    assert pairs == {(op, act) for op in ("standard", "grouped_g4") for act in ("relu", "silu")}


# --- operators -----------------------------------------------------------------------------------------------------------


def test_offspring_stay_inside_the_slice_and_are_never_copies_of_a_parent() -> None:
    parents = samples(ADMITTED_SLICE, seed=3, n=600)
    bf16_children = 0
    for seed, (a, b) in enumerate(zip(parents[0::2], parents[1::2])):
        c1, c2, info = make_offspring(a, b, GeneRng(seed + 900), ADMITTED_SLICE)
        for child in (c1, c2):
            assert ADMITTED_SLICE.contains(child)
            assert child not in (a, b)
            bf16_children += any(
                p == "bf16" for p in (child.stem_precision, child.head_precision, *(x.precision for x in child.blocks))
            )
        assert info["operators"]
    assert bf16_children > 0


def test_the_raw_operators_can_leave_the_slice_and_pin_brings_the_children_back() -> None:
    parents = samples(ADMITTED_SLICE, seed=4, n=400)
    outside = 0
    for seed, (a, b) in enumerate(zip(parents[0::2], parents[1::2])):
        for child in crossover(a, b, GeneRng(seed), SPACE_V1):
            outside += not ADMITTED_SLICE.contains(child)
            assert ADMITTED_SLICE.contains(ADMITTED_SLICE.pin(child))
        mutated = mutate(a, GeneRng(seed + 1), SPACE_V1, restricted=ADMITTED_SLICE.restricted_domains())
        assert ADMITTED_SLICE.contains(ADMITTED_SLICE.pin(mutated))
    assert outside > 0  # the scope does bite: a mixed activation and operation makes some children invalid


def test_a_genome_prepared_in_the_slice_has_the_slice_identity_and_the_outside_one_is_refused() -> None:
    good = samples(ADMITTED_SLICE, seed=7, n=1)[0]
    record = prepare_in_slice(good, ADMITTED_SLICE, context(ADMITTED_SLICE)).record
    assert record.space_hash == ADMITTED_SLICE_HASH
    bad = with_regions(stem="bf16", activation="hardswish")
    with pytest.raises(ValueError, match="outside the slice"):
        prepare_in_slice(bad, ADMITTED_SLICE, context(ADMITTED_SLICE))  # type: ignore[arg-type]


# --- the architecture slice ---------------------------------------------------------------------------------------------


def test_every_architecture_genome_is_inside_the_admitted_slice() -> None:
    for genome in samples(ARCHITECTURE_SLICE, seed=1, n=600):
        assert ARCHITECTURE_SLICE.contains(genome) and ADMITTED_SLICE.contains(genome)
    parents = samples(ARCHITECTURE_SLICE, seed=2, n=300)
    for seed, (a, b) in enumerate(zip(parents[0::2], parents[1::2])):
        for child in make_offspring(a, b, GeneRng(seed), ARCHITECTURE_SLICE)[:2]:
            assert ARCHITECTURE_SLICE.contains(child) and ADMITTED_SLICE.contains(child)


def test_the_architecture_slice_pins_exactly_the_representation_genes() -> None:
    pinned = {n for n, _ in ARCHITECTURE_SLICE.pinned_global} | {
        f"block.{n}" for n, _ in ARCHITECTURE_SLICE.pinned_block
    }
    architecture = set(ARCHITECTURE_GLOBAL) | {f"block.{n}" for n in ARCHITECTURE_BLOCK}
    every = set(GLOBAL_GENES) | {f"block.{n}" for n, _ in SPACE_V1.block_domains()}
    assert pinned | architecture == every and not pinned & architecture
    assert slice_frozen_genes(ARCHITECTURE_SLICE) == frozenset(pinned)


def test_architecture_genomes_run_in_fp32_without_pruning_and_vary_in_the_architecture() -> None:
    genomes = samples(ARCHITECTURE_SLICE, seed=3, n=200)
    for genome in genomes:
        assert genome.stem_precision == genome.head_precision == "fp32"
        assert all(b.precision == "fp32" and b.prune_pct == 0 for b in genome.blocks)
        assert (genome.int8_mode, genome.weight_granularity, genome.calibration) == ("ptq", "per_channel", "max")
    assert len({g.activation for g in genomes}) == 3 and len({g.n_blocks for g in genomes}) > 3
    assert len({b.op for g in genomes for b in g.blocks}) == 5


def test_the_sequential_method_accepts_the_pair() -> None:
    assert representation_is_searchable(ADMITTED_SLICE) and not representation_is_searchable(ARCHITECTURE_SLICE)
    rng = GeneRng(11)
    architecture, representation = ARCHITECTURE_SLICE.sample(rng), ADMITTED_SLICE.sample(rng)
    grafted = graft(architecture, representation)
    # the architecture genes (the activation, the operations) may clash with the BF16 of the representation;
    # the sequential runner pins after grafting, which puts the clash back inside the slice
    assert ADMITTED_SLICE.contains(ADMITTED_SLICE.pin(grafted))
