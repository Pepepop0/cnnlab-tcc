"""Canonical JSON, phenotype/cache-context hashes and the shared GenomeRecord."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import GenomeRecord
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import BlockGenes, Genome
from cnnlab.tcc.science.hashing import canonical_json, digest, is_sha256_hex
from cnnlab.tcc.science.record import (
    CacheContext,
    cache_context_hash,
    phenotype_hash,
    prepare_genome,
)
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1

SPACE_HASH = SPACE_V1.space_hash()


def context(**overrides: Any) -> CacheContext:
    fields: dict[str, Any] = dict(
        split_fingerprint="split-45k-5k-cal1024-v1",
        train_seed=101,
        calibration_seed=202,
        train_policy={"epochs": 5, "batch": 128, "optimizer": "adamw", "schedule": "cosine+1warmup"},
        pruning_policy={"fine_tune_epochs": 1},
        quantization_policy={"qat_epochs": 1, "qdq": "explicit-symmetric"},
        code_version="b3612d3",
        backend={"torch": "2.13.0+cu130", "tensorrt": None},
        builder_config={"strongly_typed": True},
        space_hash=SPACE_HASH,
    )
    fields.update(overrides)
    return CacheContext(**fields)


def reference_genome() -> Genome:
    blocks = (
        BlockGenes(32, "standard", 3, 2, False, "maxpool", "fp16", 0),
        BlockGenes(64, "dw_separable", 5, 1, True, "conv_s2", "int8", 25),
        BlockGenes(64, "mbconv_e3", 3, 1, True, "none", "bf16", 50),
        BlockGenes(128, "grouped_g4", 7, 3, False, "avgpool", "fp32", 0),
        BlockGenes(256, "mbconv_e6", 3, 1, True, "maxpool", "fp32", 0),  # latent
        BlockGenes(16, "standard", 3, 1, False, "none", "fp32", 0),  # latent
    )
    return Genome(
        4, 32, "silu", "gap_fc256_linear", "fp32", "fp32", "ptq", "per_channel", "entropy", -3.0, -4.0, 0.2, blocks
    )


# Pinned: a change here alters cache keys of stored candidates.
REFERENCE_PHENOTYPE = "554dd6bf334a249dec977eb1d5df22268c35d5b72a6010e34827ce53a3d035e5"


def test_canonical_json_is_order_and_container_independent() -> None:
    a = {"b": [1, 2.5, {"y": True, "x": None}], "a": "z"}
    b = {"a": "z", "b": (1, 2.5, {"x": None, "y": True})}
    assert canonical_json(a) == canonical_json(b) == '{"a":"z","b":[1,2.5,{"x":null,"y":true}]}'
    assert canonical_json({"v": -0.0}) == canonical_json({"v": 0.0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "x"}, {"p": Path("x")}, {"s": {1, 2}}, b"bytes"])
def test_canonical_json_rejects_non_canonical_values(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json(bad)


def test_digest_is_domain_separated() -> None:
    assert digest("a", {"x": 1}) != digest("b", {"x": 1})
    assert is_sha256_hex(digest("a", {"x": 1}))


def test_reference_phenotype_hash_is_pinned() -> None:
    assert phenotype_hash(express(reference_genome(), SPACE_V1).genome) == REFERENCE_PHENOTYPE


def test_prepare_genome_fills_the_contract_record() -> None:
    prepared = prepare_genome(reference_genome(), SPACE_V1, context())
    record = prepared.record
    assert isinstance(record, GenomeRecord)
    assert record.space_hash == SPACE_HASH
    assert record.raw_genome == reference_genome().to_dict()
    assert record.expressed_genome == prepared.expression.genome.to_dict()
    assert len(record.expressed_genome["blocks"]) == 4
    assert is_sha256_hex(record.phenotype_hash) and is_sha256_hex(record.cache_context_hash)
    assert all(isinstance(line, str) for line in record.repair_log)
    canonical_json(record.raw_genome)
    canonical_json(record.expressed_genome)


def test_record_survives_json_roundtrip_with_identical_hashes() -> None:
    rng = GeneRng(31)
    ctx = context()
    for _ in range(200):
        g = sample_genome(SPACE_V1, rng)
        record = prepare_genome(g, SPACE_V1, ctx).record
        stored = json.loads(json.dumps(record.raw_genome))  # as a storage layer would keep it
        again = prepare_genome(Genome.from_dict(stored), SPACE_V1, ctx).record
        assert again == record


def test_same_expression_same_context_same_hashes_despite_latent_differences() -> None:
    g = reference_genome()
    other_latent = dataclasses.replace(g, blocks=g.blocks[:4] + g.blocks[4:][::-1])
    a = prepare_genome(g, SPACE_V1, context()).record
    b = prepare_genome(other_latent, SPACE_V1, context()).record
    assert a.raw_genome != b.raw_genome
    assert (a.phenotype_hash, a.cache_context_hash) == (b.phenotype_hash, b.cache_context_hash)


@pytest.mark.parametrize(
    "overrides",
    [
        {"split_fingerprint": "split-45k-5k-cal1024-v2"},
        {"train_seed": 102},
        {"calibration_seed": 203},
        {"train_policy": {"epochs": 8, "batch": 128, "optimizer": "adamw", "schedule": "cosine+1warmup"}},
        {"pruning_policy": {"fine_tune_epochs": 2}},
        {"quantization_policy": {"qat_epochs": 2, "qdq": "explicit-symmetric"}},
        {"code_version": "deadbee"},
        {"backend": {"torch": "2.13.0+cu130", "tensorrt": "11.3.0.99"}},
        {"builder_config": {"strongly_typed": True, "opt_level": 5}},
    ],
)
def test_every_context_field_changes_only_the_cache_hash(overrides: dict[str, Any]) -> None:
    base = prepare_genome(reference_genome(), SPACE_V1, context()).record
    changed = prepare_genome(reference_genome(), SPACE_V1, context(**overrides)).record
    assert changed.phenotype_hash == base.phenotype_hash
    assert changed.cache_context_hash != base.cache_context_hash


def test_context_policy_key_order_does_not_matter() -> None:
    a = context(train_policy={"epochs": 5, "batch": 128})
    b = context(train_policy={"batch": 128, "epochs": 5})
    phenotype = "a" * 64
    assert cache_context_hash(phenotype, a) == cache_context_hash(phenotype, b)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_seed": -1},
        {"train_seed": True},
        {"calibration_seed": 1.0},
        {"split_fingerprint": ""},
        {"code_version": ""},
        {"space_hash": "space-v1"},
        {"train_policy": {"epochs": float("nan")}},
        {"backend": {"path": Path("C:/x")}},
    ],
)
def test_malformed_context_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        context(**overrides)


def test_context_for_another_space_is_rejected() -> None:
    with pytest.raises(ValueError):
        prepare_genome(reference_genome(), SPACE_V1, context(space_hash="f" * 64))


def test_cache_context_hash_requires_a_phenotype_digest() -> None:
    with pytest.raises(ValueError):
        cache_context_hash("not-a-hash", context())
