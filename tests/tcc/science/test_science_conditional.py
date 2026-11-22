"""Conditional-technique machinery, exercised with *synthetic* admissions.

These tests do not admit FP8, INT4/FP4 or 2:4 for the TCC: they only check that,
once a real admission exists, sampling stays inside the admitted scope and
repair falls back deterministically. ``SPACE_V1`` stays without conditionals.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from cnnlab.tcc.science.expression import RULE_CONDITIONAL_SCOPE, express
from cnnlab.tcc.science.genome import GenomeDomainError, validate_expressed, validate_genome
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1, AdmissionEvidence, SearchSpace

SYNTHETIC_SHA = "e" * 64


def synthetic(technique: str, scope: tuple[str, ...]) -> AdmissionEvidence:
    return AdmissionEvidence(technique, scope, SYNTHETIC_SHA, True, True, 2)


FP8_STANDARD = SearchSpace(admissions=(synthetic("fp8", ("standard",)),))
INT4_MBCONV = SearchSpace(admissions=(synthetic("int4_weights", ("mbconv_e3", "mbconv_e6")),))
SPARSE = SearchSpace(admissions=(synthetic("sparse_2_4", ("standard", "grouped_g4")),))


def test_sampler_places_conditional_values_only_inside_scope() -> None:
    rng = GeneRng(1)
    placed = 0
    for _ in range(2000):
        g = sample_genome(FP8_STANDARD, rng)
        validate_genome(g, FP8_STANDARD)
        assert express(g, FP8_STANDARD).repairs == ()
        assert g.stem_precision != "fp8" and g.head_precision != "fp8"
        for b in g.blocks:
            if b.precision == "fp8":
                assert b.op == "standard"
                placed += 1
    assert placed > 0


def test_op_change_out_of_scope_falls_back_deterministically() -> None:
    g = sample_genome(FP8_STANDARD, GeneRng(2))
    blocks = list(g.blocks)
    blocks[0] = dataclasses.replace(blocks[0], op="dw_separable", precision="fp8")
    g = dataclasses.replace(g, blocks=tuple(blocks))
    expression = express(g, FP8_STANDARD)
    assert expression.genome.blocks[0].precision == "fp16"
    first = json.loads(expression.repair_log[0])
    assert (first["rule"], first["path"], first["before"], first["after"]) == (
        RULE_CONDITIONAL_SCOPE,
        "blocks[0].precision",
        "fp8",
        "fp16",
    )
    validate_expressed(expression.genome, FP8_STANDARD)
    assert g.blocks[0].precision == "fp8"  # raw untouched


def test_weight_only_fallback_expresses_int8_genes() -> None:
    g = sample_genome(INT4_MBCONV, GeneRng(3))
    blocks = tuple(dataclasses.replace(b, precision="fp32") for b in g.blocks)
    blocks = (dataclasses.replace(blocks[0], op="standard", precision="int4w"),) + blocks[1:]
    g = dataclasses.replace(g, stem_precision="fp32", head_precision="fp32", int8_mode="ptq", blocks=blocks)
    expressed = express(g, INT4_MBCONV).genome
    assert expressed.blocks[0].precision == "int8"
    assert expressed.int8_mode == "ptq" and expressed.calibration is not None


def test_sparsity_gene_exists_only_when_admitted() -> None:
    g = sample_genome(SPARSE, GeneRng(4))
    assert all(b.sparsity in ("dense", "2:4") for b in g.blocks)
    with pytest.raises(GenomeDomainError):
        validate_genome(g, SPACE_V1)  # base space has no sparsity gene
    base = sample_genome(SPACE_V1, GeneRng(4))
    assert all(b.sparsity is None for b in base.blocks)
    assert all("sparsity" not in b for b in base.to_dict()["blocks"])


def test_sparsity_on_depthwise_is_repaired_to_dense() -> None:
    g = sample_genome(SPARSE, GeneRng(5))
    blocks = list(g.blocks)
    blocks[0] = dataclasses.replace(blocks[0], op="dw_separable", sparsity="2:4")
    expression = express(dataclasses.replace(g, blocks=tuple(blocks)), SPARSE)
    assert expression.genome.blocks[0].sparsity == "dense"
    assert any(json.loads(line)["path"] == "blocks[0].sparsity" for line in expression.repair_log)


def test_sampled_sparsity_respects_scope() -> None:
    rng = GeneRng(6)
    for _ in range(1000):
        for b in sample_genome(SPARSE, rng).blocks:
            if b.sparsity == "2:4":
                assert b.op in ("standard", "grouped_g4")
