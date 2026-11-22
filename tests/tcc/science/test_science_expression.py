"""Repair/expression invariants (TCC-02 section 3) and the derived network plan."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import pytest

from cnnlab.tcc.science.expression import (
    RULE_MAX_REDUCTIONS,
    express,
    plan_network,
    pruned_width,
    repaired_genome,
)
from cnnlab.tcc.science.genome import BlockGenes, Genome, validate_expressed
from cnnlab.tcc.science.record import phenotype_hash
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import (
    INT8_GENES,
    MAX_SPATIAL_REDUCTIONS,
    SPACE_V1,
    Choice,
    Interval,
)

N = 1500


def arbitrary_genome(rng: GeneRng) -> Genome:
    """Every gene drawn independently from its raw domain: models operator output that may need repair."""
    genes: dict[str, Any] = {}
    for name, domain in SPACE_V1.global_domains():
        genes[name] = (
            rng.uniform(domain.low, domain.high) if isinstance(domain, Interval) else rng.choice(domain.values)
        )
    blocks = []
    for _ in range(6):
        block_genes = {
            name: rng.choice(dom.values) for name, dom in SPACE_V1.block_domains() if isinstance(dom, Choice)
        }
        blocks.append(BlockGenes(**block_genes))
    return Genome(**genes, blocks=tuple(blocks))


def arbitrary(seed: int, n: int = N) -> list[Genome]:
    rng = GeneRng(seed)
    return [arbitrary_genome(rng) for _ in range(n)]


def with_block(g: Genome, slot: int, **changes: Any) -> Genome:
    blocks = list(g.blocks)
    blocks[slot] = dataclasses.replace(blocks[slot], **changes)
    return dataclasses.replace(g, blocks=tuple(blocks))


def all_reducing(n_blocks: int) -> Genome:
    g = sample_genome(SPACE_V1, GeneRng(4))
    blocks = tuple(dataclasses.replace(b, downsample="maxpool") for b in g.blocks)
    return dataclasses.replace(g, n_blocks=n_blocks, blocks=blocks)


# --- purity and determinism --------------------------------------------------


def test_repair_never_modifies_the_raw_genome() -> None:
    repaired_any = False
    for g in arbitrary(1):
        snapshot = copy.deepcopy(g.to_dict())
        frozen_copy = dataclasses.replace(g)
        expression = express(g, SPACE_V1)
        repaired_genome(g, SPACE_V1)
        assert g.to_dict() == snapshot
        assert g == frozen_copy
        repaired_any |= bool(expression.repairs)
    assert repaired_any, "fixture must exercise at least one repair"


def test_expression_is_deterministic() -> None:
    for g in arbitrary(2, 300):
        a, b = express(g, SPACE_V1), express(Genome.from_dict(json.loads(json.dumps(g.to_dict()))), SPACE_V1)
        assert a == b
        assert a.repair_log == b.repair_log


def test_expression_is_idempotent() -> None:
    for g in arbitrary(3):
        first = express(g, SPACE_V1)
        second = express(repaired_genome(g, SPACE_V1), SPACE_V1)
        assert second.repairs == ()
        assert second.genome == first.genome


def test_every_expression_satisfies_post_repair_invariants() -> None:
    for g in arbitrary(4):
        expression = express(g, SPACE_V1)
        assert expression.valid, expression.invalid_reason
        validate_expressed(expression.genome, SPACE_V1)
        assert len(expression.genome.blocks) == g.n_blocks


# --- R1: spatial reductions --------------------------------------------------


def test_excess_reductions_are_removed_from_the_last_block() -> None:
    g = all_reducing(6)
    expression = express(g, SPACE_V1)
    downsample = [b.downsample for b in expression.genome.blocks]
    assert downsample == ["maxpool"] * 4 + ["none", "none"]
    logs = [json.loads(line) for line in expression.repair_log]
    assert [(r["rule"], r["path"], r["before"], r["after"]) for r in logs] == [
        (RULE_MAX_REDUCTIONS, "blocks[5].downsample", "maxpool", "none"),
        (RULE_MAX_REDUCTIONS, "blocks[4].downsample", "maxpool", "none"),
    ]
    resolutions = [b.out_resolution for b in plan_network(expression.genome).blocks]
    assert resolutions == [16, 8, 4, 2, 2, 2]


def test_latent_reductions_are_not_repaired_nor_counted() -> None:
    expression = express(all_reducing(2), SPACE_V1)
    assert expression.repairs == ()
    assert [b.downsample for b in expression.genome.blocks] == ["maxpool", "maxpool"]


def test_resolution_never_drops_below_two() -> None:
    for g in arbitrary(5):
        plan = plan_network(express(g, SPACE_V1).genome)
        assert plan.final_resolution >= 2
        assert sum(b.out_resolution != b.in_resolution for b in plan.blocks) <= MAX_SPATIAL_REDUCTIONS


# --- phenotype hash: latent vs expressed genes -------------------------------


def test_latent_slots_do_not_change_the_phenotype() -> None:
    rng = GeneRng(6)
    for g in arbitrary(7, 500):
        other = arbitrary_genome(rng)
        mixed = dataclasses.replace(g, blocks=g.blocks[: g.n_blocks] + other.blocks[g.n_blocks :])
        assert mixed.to_dict() != g.to_dict() or g.n_blocks == 6
        assert phenotype_hash(express(mixed, SPACE_V1).genome) == phenotype_hash(express(g, SPACE_V1).genome)


def fp32_everywhere(g: Genome) -> Genome:
    blocks = tuple(dataclasses.replace(b, precision="fp32") for b in g.blocks)
    return dataclasses.replace(g, stem_precision="fp32", head_precision="fp32", blocks=blocks)


def test_quantization_genes_are_latent_without_int8() -> None:
    for g in arbitrary(8, 300):
        g = fp32_everywhere(g)
        base = phenotype_hash(express(g, SPACE_V1).genome)
        for name, values in (
            ("int8_mode", ("ptq", "qat")),
            ("weight_granularity", ("per_tensor", "per_channel")),
            ("calibration", ("max", "entropy", "percentile_99_99")),
        ):
            for value in values:
                variant = express(dataclasses.replace(g, **{name: value}), SPACE_V1).genome
                assert phenotype_hash(variant) == base
                assert name not in variant.to_dict()


def test_int8_in_latent_slot_does_not_express_quantization_genes() -> None:
    g = fp32_everywhere(sample_genome(SPACE_V1, GeneRng(9)))
    g = dataclasses.replace(g, n_blocks=2)
    latent_int8 = with_block(g, 4, precision="int8")
    expressed = express(latent_int8, SPACE_V1).genome
    assert all(getattr(expressed, name) is None for name in INT8_GENES)
    assert phenotype_hash(expressed) == phenotype_hash(express(g, SPACE_V1).genome)


@pytest.mark.parametrize("where", ["stem", "head", "block"])
def test_calibration_expressed_only_in_ptq(where: str) -> None:
    g = fp32_everywhere(sample_genome(SPACE_V1, GeneRng(10)))
    if where == "stem":
        g = dataclasses.replace(g, stem_precision="int8")
    elif where == "head":
        g = dataclasses.replace(g, head_precision="int8")
    else:
        g = with_block(g, 0, precision="int8")

    def h(**changes: Any) -> str:
        return phenotype_hash(express(dataclasses.replace(g, **changes), SPACE_V1).genome)

    assert h(int8_mode="qat", calibration="max") == h(int8_mode="qat", calibration="entropy")
    assert h(int8_mode="ptq", calibration="max") != h(int8_mode="ptq", calibration="entropy")
    assert h(int8_mode="qat", weight_granularity="per_tensor") != h(int8_mode="qat", weight_granularity="per_channel")
    assert h(int8_mode="ptq") != h(int8_mode="qat")


def test_hash_equality_iff_expressed_equality_under_single_gene_changes() -> None:
    """Change one active gene at a time: the hash changes exactly when the expression does."""
    changed = unchanged = 0
    for g in arbitrary(11, 60):
        base = express(g, SPACE_V1).genome
        base_hash = phenotype_hash(base)
        variants: list[Genome] = []
        for name, domain in SPACE_V1.global_domains():
            if isinstance(domain, Choice):
                variants += [dataclasses.replace(g, **{name: v}) for v in domain.values if v != getattr(g, name)]
            else:
                variants.append(dataclasses.replace(g, **{name: (domain.low + domain.high) / 2}))
        for slot in range(g.n_blocks):
            for name, domain in SPACE_V1.block_domains():
                assert isinstance(domain, Choice)
                variants += [
                    with_block(g, slot, **{name: v}) for v in domain.values if v != getattr(g.blocks[slot], name)
                ]
        for variant in variants:
            expressed = express(variant, SPACE_V1).genome
            same = expressed == base
            assert (phenotype_hash(expressed) == base_hash) == same
            changed += not same
            unchanged += same
    assert changed > 1000 and unchanged > 0  # both branches exercised


# --- derived plan (R2, R3 and shortcuts) --------------------------------------


EXPECTED_PRUNED = {
    16: (16, 16, 16),
    32: (32, 32, 16),
    48: (48, 32, 32),
    64: (64, 48, 32),
    96: (96, 80, 48),
    128: (128, 96, 64),
    192: (192, 144, 96),
    256: (256, 192, 128),
}


def test_pruned_width_table() -> None:
    for channels, expected in EXPECTED_PRUNED.items():
        assert tuple(pruned_width(channels, p) for p in (0, 25, 50)) == expected


def test_pruned_width_properties() -> None:
    for channels in EXPECTED_PRUNED:
        widths = [pruned_width(channels, p) for p in (0, 25, 50)]
        assert widths[0] == channels
        assert all(w % 16 == 0 and 16 <= w <= channels for w in widths)
        assert widths == sorted(widths, reverse=True)
    with pytest.raises(ValueError):
        pruned_width(24, 25)


def test_plan_shortcuts_groups_and_widths() -> None:
    for g in arbitrary(12):
        expressed = express(g, SPACE_V1).genome
        plan = plan_network(expressed)
        in_channels = expressed.stem_channels
        for b, genes in zip(plan.blocks, expressed.blocks):
            assert b.in_channels == in_channels
            assert b.out_channels == genes.channels
            if not genes.residual:
                assert b.shortcut == "none"
            elif b.in_channels == b.out_channels and b.in_resolution == b.out_resolution:
                assert b.shortcut == "identity"
            else:
                assert b.shortcut == "projection"
            if genes.op == "grouped_g4":
                assert b.groups == 4
                assert b.in_channels % 4 == 0 and b.out_channels % 4 == 0 and b.nominal_kept_channels % 4 == 0
            if genes.op.startswith("mbconv"):
                assert b.hidden_channels == b.in_channels * int(genes.op[-1])
            in_channels = genes.channels
        assert plan.head_in_channels == plan.blocks[-1].nominal_kept_channels
        assert plan.num_classes == 100
        assert set(plan.identity_coupled_blocks) == {b.index for b in plan.blocks if b.shortcut == "identity"}
