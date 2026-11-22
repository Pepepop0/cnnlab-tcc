"""Structure, strict parsing and domain validation of raw and expressed genomes."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import (
    BlockGenes,
    ExpressedGenome,
    Genome,
    GenomeDomainError,
    GenomeStructureError,
    validate_expressed,
    validate_genome,
)
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1


def block(**overrides: Any) -> BlockGenes:
    genes: dict[str, Any] = dict(
        channels=32,
        op="standard",
        kernel=3,
        n_convs=1,
        residual=False,
        downsample="none",
        precision="fp32",
        prune_pct=0,
    )
    genes.update(overrides)
    return BlockGenes(**genes)


def genome(blocks: tuple[BlockGenes, ...] | None = None, **overrides: Any) -> Genome:
    genes: dict[str, Any] = dict(
        n_blocks=4,
        stem_channels=32,
        activation="relu",
        head="gap_linear",
        stem_precision="fp32",
        head_precision="fp32",
        int8_mode="ptq",
        weight_granularity="per_channel",
        calibration="max",
        log10_lr=-3.0,
        log10_wd=-4.5,
        head_dropout=0.1,
        blocks=blocks if blocks is not None else tuple(block() for _ in range(6)),
    )
    genes.update(overrides)
    return Genome(**genes)


def test_roundtrip_through_json_is_exact() -> None:
    rng = GeneRng(11)
    for _ in range(300):
        g = sample_genome(SPACE_V1, rng)
        data = json.loads(json.dumps(g.to_dict()))
        assert Genome.from_dict(data) == g
        e = express(g, SPACE_V1).genome
        assert ExpressedGenome.from_dict(json.loads(json.dumps(e.to_dict()))) == e


def test_genome_is_frozen_and_to_dict_is_a_copy() -> None:
    g = genome()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.n_blocks = 5  # type: ignore[misc]
    data = g.to_dict()
    data["blocks"][0]["channels"] = 256
    data["n_blocks"] = 6
    assert g.blocks[0].channels == 32 and g.n_blocks == 4


def test_raw_genome_needs_exactly_six_slots() -> None:
    with pytest.raises(GenomeStructureError):
        genome(blocks=tuple(block() for _ in range(5)))
    with pytest.raises(GenomeStructureError):
        genome(blocks=tuple(block() for _ in range(7)))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("head"),
        lambda d: d.update(extra=1),
        lambda d: d.update(schema="tcc-genome/v0"),
        lambda d: d["blocks"][2].pop("kernel"),
        lambda d: d["blocks"][2].update(width=3),
        lambda d: d.update(blocks="not-a-list"),
    ],
)
def test_from_dict_rejects_malformed_structure(mutate: Any) -> None:
    data = genome().to_dict()
    mutate(data)
    with pytest.raises(GenomeStructureError):
        Genome.from_dict(data)


@pytest.mark.parametrize(
    ("where", "field", "value"),
    [
        ("global", "n_blocks", True),
        ("global", "n_blocks", 4.0),
        ("global", "activation", 1),
        ("global", "log10_lr", True),
        ("global", "log10_lr", "0.001"),
        ("global", "log10_lr", float("nan")),
        ("global", "head_dropout", float("inf")),
        ("block", "channels", 32.0),
        ("block", "channels", True),
        ("block", "residual", 1),
        ("block", "op", None),
    ],
)
def test_types_are_strict(where: str, field: str, value: Any) -> None:
    with pytest.raises(GenomeStructureError):
        if where == "global":
            genome(**{field: value})
        else:
            block(**{field: value})


def test_continuous_genes_normalize_int_and_negative_zero() -> None:
    a = genome(log10_lr=-3, head_dropout=-0.0)
    b = genome(log10_lr=-3.0, head_dropout=0.0)
    assert a == b
    assert type(a.log10_lr) is float
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"n_blocks": 1}, "n_blocks"),
        ({"n_blocks": 7}, "n_blocks"),
        ({"stem_channels": 48}, "stem_channels"),
        ({"activation": "gelu"}, "activation"),
        ({"head": "gap_fc512_linear"}, "head"),
        ({"stem_precision": "fp8"}, "stem_precision"),
        ({"head_precision": "int4w"}, "head_precision"),
        ({"int8_mode": "dynamic"}, "int8_mode"),
        ({"weight_granularity": "per_group"}, "weight_granularity"),
        ({"calibration": "mse"}, "calibration"),
        ({"log10_lr": -4.5}, "log10_lr"),
        ({"log10_lr": -1.0}, "log10_lr"),
        ({"log10_wd": -2.9}, "log10_wd"),
        ({"head_dropout": 0.51}, "head_dropout"),
        ({"head_dropout": -0.01}, "head_dropout"),
    ],
)
def test_global_domain_violations(overrides: dict[str, Any], path: str) -> None:
    with pytest.raises(GenomeDomainError) as info:
        validate_genome(genome(**overrides), SPACE_V1)
    assert info.value.path == path


@pytest.mark.parametrize(
    "overrides",
    [
        {"channels": 24},
        {"channels": 512},
        {"op": "grouped_g8"},
        {"kernel": 1},
        {"n_convs": 4},
        {"downsample": "stride3"},
        {"precision": "fp8"},
        {"precision": "int4w"},
        {"prune_pct": 75},
        {"sparsity": "2:4"},
        {"sparsity": "dense"},
    ],
)
@pytest.mark.parametrize("slot", [0, 5])  # slot 5 is latent when n_blocks=4
def test_block_domain_violations_including_latent_slots(overrides: dict[str, Any], slot: int) -> None:
    blocks = [block() for _ in range(6)]
    blocks[slot] = block(**overrides)
    with pytest.raises(GenomeDomainError) as info:
        validate_genome(genome(blocks=tuple(blocks)), SPACE_V1)
    assert info.value.path.startswith(f"blocks[{slot}].")


def test_domain_boundaries_are_valid() -> None:
    for g in (
        genome(log10_lr=-4.0, log10_wd=-6.0, head_dropout=0.0, n_blocks=2),
        genome(log10_lr=-2.0, log10_wd=-3.0, head_dropout=0.5, n_blocks=6),
    ):
        validate_genome(g, SPACE_V1)


def test_expressed_invariants_reject_inconsistent_quantization_genes() -> None:
    base = express(genome(), SPACE_V1).genome  # all fp32: INT8 genes latent
    assert base.int8_mode is None
    with pytest.raises(GenomeDomainError):
        validate_expressed(dataclasses.replace(base, int8_mode="ptq"), SPACE_V1)
    int8 = express(genome(stem_precision="int8", int8_mode="qat"), SPACE_V1).genome
    assert int8.calibration is None
    validate_expressed(int8, SPACE_V1)
    with pytest.raises(GenomeDomainError):
        validate_expressed(dataclasses.replace(int8, calibration="max"), SPACE_V1)


def test_expressed_invariants_reject_excess_reductions() -> None:
    expressed = express(genome(n_blocks=5), SPACE_V1).genome
    too_many = tuple(dataclasses.replace(b, downsample="maxpool") for b in expressed.blocks)
    with pytest.raises(GenomeDomainError):
        validate_expressed(dataclasses.replace(expressed, blocks=too_many), SPACE_V1)


def test_expressed_block_count_must_match_n_blocks() -> None:
    expressed = express(genome(), SPACE_V1).genome
    with pytest.raises(GenomeStructureError):
        dataclasses.replace(expressed, blocks=expressed.blocks[:2])
