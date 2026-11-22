"""CPU tests of the PyTorch builder for expressed genome v1: shape, forward, gradient, determinism.

All inputs are seeded random tensors; no CIFAR-100 split is read and nothing
here measures E/L/V/S.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from torch import nn

from cnnlab.tcc.model_pipeline.builder import (
    ACTIVATIONS,
    BUILDER_VERSION,
    Block,
    ModelBuildError,
    TCCNet,
    build_model,
    check_buildable,
    count_macs,
    count_parameters,
    fold_batchnorm,
)
from cnnlab.tcc.science.expression import NetworkPlan, express, plan_network
from cnnlab.tcc.science.genome import BlockGenes, ExpressedGenome, Genome
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import DOWNSAMPLES, OPS, SPACE_V1

GLOBALS: dict[str, Any] = {
    "stem_channels": 16,
    "activation": "relu",
    "head": "gap_linear",
    "stem_precision": "fp32",
    "head_precision": "fp32",
    "int8_mode": "ptq",
    "weight_granularity": "per_tensor",
    "calibration": "max",
    "log10_lr": -3.0,
    "log10_wd": -4.0,
    "head_dropout": 0.1,
}
BLOCK: dict[str, Any] = {
    "channels": 16,
    "op": "standard",
    "kernel": 3,
    "n_convs": 1,
    "residual": False,
    "downsample": "none",
    "precision": "fp32",
    "prune_pct": 0,
}


def expressed(blocks: list[dict[str, Any]], **globals_: Any) -> ExpressedGenome:
    """Express a genome whose active blocks are ``blocks`` (latent slots are default blocks)."""
    slots = [BlockGenes(**{**BLOCK, **b}) for b in blocks]
    slots += [BlockGenes(**BLOCK)] * (6 - len(slots))
    genome = Genome(n_blocks=len(blocks), **{**GLOBALS, **globals_}, blocks=tuple(slots))
    expression = express(genome, SPACE_V1)
    assert expression.valid
    return expression.genome


def sampled(seed: int) -> ExpressedGenome:
    return express(sample_genome(SPACE_V1, GeneRng(seed)), SPACE_V1).genome


def images(n: int, seed: int = 0) -> torch.Tensor:
    return torch.randn((n, 3, 32, 32), generator=torch.Generator().manual_seed(seed))


def block_outputs(model: TCCNet, x: torch.Tensor) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    handles = [b.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape))) for b in model.blocks]
    try:
        with torch.no_grad():
            model.eval()(x)
    finally:
        for h in handles:
            h.remove()
    return shapes


def assert_matches_plan(model: TCCNet, info: Any, genome: ExpressedGenome, batch: int = 2) -> torch.Tensor:
    x = images(batch)
    shapes = block_outputs(model, x)
    assert shapes == [(batch, *s) for s in info.block_out_shapes]
    plan = plan_network(genome)
    assert [(p.out_channels, p.out_resolution, p.out_resolution) for p in plan.blocks] == list(info.block_out_shapes)
    with torch.no_grad():
        logits = model.eval()(x)
    assert logits.shape == (batch, 100) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    return logits


# --- shape and forward ------------------------------------------------------------


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("downsample", DOWNSAMPLES)
@pytest.mark.parametrize("op", OPS)
def test_every_op_downsample_and_residual_combination(op: str, downsample: str, residual: bool) -> None:
    genome = expressed(
        [
            {"channels": 32, "op": op, "downsample": downsample, "residual": residual, "n_convs": 2},
            {"channels": 32, "op": op, "downsample": "none", "residual": residual},
        ]
    )
    model, info = build_model(genome, seed=1)
    assert_matches_plan(model, info, genome)
    expected_first = "none" if not residual else "projection"  # 16 -> 32 channels always projects
    expected_second = "none" if not residual else "identity"
    assert info.shortcuts == (expected_first, expected_second)


@pytest.mark.parametrize("n_convs", [1, 2, 3])
@pytest.mark.parametrize("kernel", [3, 5, 7])
@pytest.mark.parametrize("op", OPS)
def test_kernels_and_unit_counts(op: str, kernel: int, n_convs: int) -> None:
    genome = expressed(
        [
            {"channels": 48, "op": op, "kernel": kernel, "n_convs": n_convs, "downsample": "conv_s2"},
            {"channels": 48, "op": op, "kernel": kernel, "n_convs": n_convs, "residual": True},
        ]
    )
    model, info = build_model(genome, seed=2)
    assert_matches_plan(model, info, genome)
    spatial = [m for m in model.blocks[0].body if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1)]
    assert len(spatial) == n_convs and all(m.kernel_size == (kernel, kernel) for m in spatial)
    assert spatial[0].stride == (2, 2) and all(m.stride == (1, 1) for m in spatial[1:])


@pytest.mark.parametrize("seed", range(40))
def test_sampled_genomes_build_and_match_plan(seed: int) -> None:
    genome = sampled(seed)
    model, info = build_model(genome, seed=seed)
    assert_matches_plan(model, info, genome, batch=1)
    assert info.parameters == count_parameters(model) > 0


def test_four_reductions_reach_two_pixels() -> None:
    genome = expressed([{"downsample": d} for d in ("maxpool", "avgpool", "conv_s2", "maxpool", "none", "none")])
    model, info = build_model(genome, seed=0)
    assert [s[1] for s in info.block_out_shapes] == [16, 8, 4, 2, 2, 2]
    assert_matches_plan(model, info, genome)


# --- op internals ---------------------------------------------------------------------


def convs(module: nn.Module) -> list[nn.Conv2d]:
    return [m for m in module.modules() if isinstance(m, nn.Conv2d)]


def test_grouped_uses_four_groups_and_depthwise_uses_channels() -> None:
    model, _ = build_model(
        expressed(
            [
                {"channels": 32, "op": "grouped_g4"},
                {"channels": 64, "op": "dw_separable"},
            ]
        ),
        seed=0,
    )
    assert [c.groups for c in convs(model.blocks[0].body)] == [4]
    dw, pw = convs(model.blocks[1].body)
    assert (dw.groups, dw.in_channels, dw.out_channels) == (32, 32, 32)
    assert (pw.kernel_size, pw.in_channels, pw.out_channels) == ((1, 1), 32, 64)


@pytest.mark.parametrize(("op", "e"), [("mbconv_e3", 3), ("mbconv_e6", 6)])
def test_mbconv_expands_unit_input_and_projects_linearly(op: str, e: int) -> None:
    genome = expressed([{"channels": 32, "op": op, "n_convs": 2, "residual": True}, {"channels": 32}])
    model, _ = build_model(genome, seed=0)
    block = model.blocks[0]
    c = convs(block.body)
    # unit 1 over C_in=16, unit 2 over C_out=32 (plan_network: hidden = C_in * e for the first unit)
    assert [(k.in_channels, k.out_channels, k.groups) for k in c] == [
        (16, 16 * e, 1),
        (16 * e, 16 * e, 16 * e),
        (16 * e, 32, 1),
        (32, 32 * e, 1),
        (32 * e, 32 * e, 32 * e),
        (32 * e, 32, 1),
    ]
    assert plan_network(genome).blocks[0].hidden_channels == 16 * e
    assert isinstance(block.body[-1], nn.BatchNorm2d)  # linear projection
    assert isinstance(block.post_act, nn.Identity)


def test_residual_add_is_followed_by_activation_for_non_mbconv() -> None:
    model, _ = build_model(expressed([{"residual": True}, {}], activation="silu"), seed=0)
    block = model.blocks[0]
    assert isinstance(block.shortcut, nn.Identity)
    assert isinstance(block.post_act, nn.SiLU)
    assert isinstance(block.body[-1], nn.BatchNorm2d)  # no activation before the add


@pytest.mark.parametrize(
    ("block", "stride"),
    [({"channels": 32}, 1), ({"downsample": "maxpool"}, 2), ({"channels": 32, "downsample": "conv_s2"}, 2)],
)
def test_projection_shortcut_is_1x1_conv_bn(block: dict[str, Any], stride: int) -> None:
    model, info = build_model(expressed([{**block, "residual": True}, {}]), seed=0)
    assert info.shortcuts[0] == "projection"
    conv, bn = model.blocks[0].shortcut
    assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
    assert conv.kernel_size == (1, 1) and conv.stride == (stride, stride) and conv.bias is None


def test_no_residual_has_no_shortcut() -> None:
    model, info = build_model(expressed([{"channels": 32}, {}]), seed=0)
    assert model.blocks[0].shortcut is None and info.shortcuts[0] == "none"


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_activation_gene_is_used_everywhere(activation: str) -> None:
    model, _ = build_model(expressed([{"residual": True}, {"op": "dw_separable"}], activation=activation), seed=0)
    kinds = {type(m) for m in model.modules() if isinstance(m, (nn.ReLU, nn.SiLU, nn.Hardswish))}
    assert kinds == {type(ACTIVATIONS[activation]())}


@pytest.mark.parametrize(
    ("head", "linears"), [("gap_linear", [(64, 100)]), ("gap_fc256_linear", [(64, 256), (256, 100)])]
)
def test_head_variants_and_dropout(head: str, linears: list[tuple[int, int]]) -> None:
    model, _ = build_model(expressed([{"channels": 64}, {"channels": 64}], head=head, head_dropout=0.37), seed=0)
    assert [(m.in_features, m.out_features) for m in model.head.modules() if isinstance(m, nn.Linear)] == linears
    (dropout,) = [m for m in model.head.modules() if isinstance(m, nn.Dropout)]
    assert dropout.p == pytest.approx(0.37)


# --- deferred genes: never silently applied ----------------------------------------------


def test_pruning_and_precision_are_reported_not_applied() -> None:
    genome = expressed(
        [{"channels": 64, "prune_pct": 50, "precision": "int8"}, {"channels": 32, "precision": "bf16"}],
        stem_precision="fp16",
    )
    model, info = build_model(genome, seed=0)
    assert convs(model.blocks[0].body)[-1].out_channels == 64  # unpruned width
    assert info.unpruned
    assert "blocks[0].prune_pct" in info.deferred_representation
    assert "blocks[1].prune_pct" not in info.deferred_representation
    assert {
        "stem_precision",
        "head_precision",
        "int8_mode",
        "weight_granularity",
        "calibration",
        "blocks[0].precision",
        "blocks[1].precision",
    } <= set(info.deferred_representation)
    assert info.deferred_training == ("log10_lr", "log10_wd")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert info.builder_version == BUILDER_VERSION


def test_latent_int8_genes_are_not_reported() -> None:
    _, info = build_model(expressed([{}, {}]), seed=0)
    assert not {"int8_mode", "weight_granularity", "calibration"} & set(info.deferred_representation)


# --- gradient -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_training_step_gives_finite_gradients_everywhere(seed: int) -> None:
    genome = sampled(1000 + seed)
    model, _ = build_model(genome, seed=seed)
    model.train()
    x = images(4, seed)
    y = torch.arange(4) * 7 % 100
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert model.stem[0].weight.grad.abs().sum() > 0
    assert list(model.head.modules())[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("op", OPS)
def test_one_sgd_step_reduces_loss_on_a_fixed_batch(op: str) -> None:
    model, _ = build_model(
        expressed([{"channels": 32, "op": op, "residual": True}, {"channels": 32}], head_dropout=0.0), seed=3
    )
    model.train()
    x, y = images(8, 5), torch.arange(8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


# --- determinism ------------------------------------------------------------------------


def state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_same_seed_same_weights_and_logits(seed: int) -> None:
    genome = sampled(seed)
    a, info_a = build_model(genome, seed=seed)
    b, info_b = build_model(genome, seed=seed)
    sa, sb = state(a), state(b)
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert info_a == info_b
    x = images(2)
    with torch.no_grad():
        assert torch.equal(a.eval()(x), b.eval()(x))


def test_different_seed_changes_weights() -> None:
    genome = sampled(4)
    a, _ = build_model(genome, seed=1)
    b, _ = build_model(genome, seed=2)
    assert not torch.equal(a.stem[0].weight, b.stem[0].weight)


def test_build_leaves_global_rng_untouched() -> None:
    torch.manual_seed(99)
    before = torch.get_rng_state()
    build_model(sampled(5), seed=11)
    assert torch.equal(torch.get_rng_state(), before)


def test_same_phenotype_from_different_latent_genes_builds_identical_model() -> None:
    a = expressed([{"channels": 32}, {}])
    genome_b = Genome(
        n_blocks=2,
        **GLOBALS,
        blocks=(
            BlockGenes(**{**BLOCK, "channels": 32}),
            BlockGenes(**BLOCK),
            *[BlockGenes(**{**BLOCK, "op": "mbconv_e6"})] * 4,
        ),
    )
    b = express(genome_b, SPACE_V1).genome
    assert a == b
    ma, _ = build_model(a, seed=0)
    mb, _ = build_model(b, seed=0)
    assert all(torch.equal(x, y) for x, y in zip(state(ma).values(), state(mb).values()))


# --- auxiliary metrics and invalid input ------------------------------------------------------


def test_macs_match_hand_count() -> None:
    model, _ = build_model(expressed([{}, {}]), seed=0)
    stem = 32 * 32 * 16 * 3 * 9
    block = 32 * 32 * 16 * 16 * 9
    assert count_macs(model) == stem + 2 * block + 16 * 100
    assert model.training  # restored


def test_build_rejects_raw_genome_and_bad_seed() -> None:
    genome = Genome(n_blocks=2, **GLOBALS, blocks=tuple(BlockGenes(**BLOCK) for _ in range(6)))
    with pytest.raises(TypeError):
        build_model(genome, seed=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_model(expressed([{}, {}]), seed=-1)
    with pytest.raises(ValueError):
        build_model(expressed([{}, {}]), seed=True)  # type: ignore[arg-type]


def test_unbuildable_resolution_raises_model_build_error() -> None:
    # Five reductions cannot come out of express(); a genome read from storage might.
    blocks = tuple(BlockGenes(**{**BLOCK, "downsample": "maxpool"}) for _ in range(5))
    bad = ExpressedGenome(
        n_blocks=5, **{**GLOBALS, "int8_mode": None, "weight_granularity": None, "calibration": None}, blocks=blocks
    )
    with pytest.raises(ModelBuildError) as err:
        build_model(bad, seed=0)
    assert err.value.code == "R1_resolution"


def test_grouped_divisibility_is_checked() -> None:
    plan = plan_network(expressed([{"op": "grouped_g4"}, {}]))
    broken = dataclasses.replace(plan.blocks[0], in_channels=18)
    with pytest.raises(ModelBuildError) as err:
        check_buildable(NetworkPlan(16, (broken,), 16, None, 100))
    assert err.value.code == "R2_grouped_divisibility"


def test_block_is_exported_for_typed_regions() -> None:
    model, _ = build_model(sampled(8), seed=0)
    assert isinstance(model, TCCNet)
    assert all(isinstance(b, Block) for b in model.blocks)


@pytest.mark.parametrize("seed", range(6))
def test_fold_batchnorm_preserves_eval_logits_and_removes_bn(seed: int) -> None:
    model, _ = build_model(sampled(2000 + seed), seed=seed)
    model.train()
    with torch.no_grad():  # move BN statistics away from their init so folding is observable
        for _ in range(2):
            model(images(4, seed))
    folded = fold_batchnorm(model)
    assert model.training  # the original is not touched
    assert not any(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == sum(
        isinstance(m, nn.Conv2d) for m in model.modules()
    )
    x = images(2, 100 + seed)
    with torch.no_grad():
        torch.testing.assert_close(folded(x), model.eval()(x), rtol=1e-4, atol=1e-4)


def test_builder_smoke_genomes_cover_every_op_and_build_on_cpu() -> None:
    from cnnlab.tcc.model_pipeline.builder_smoke import LAYOUTS, smoke_genomes

    genomes = smoke_genomes(sampled=2)
    assert [label for label, _ in genomes[: len(OPS)]] == [f"op:{op}" for op in OPS]
    assert {layout for layout, _, _ in LAYOUTS} == {"fp32_folded", "bf16_folded", "bf16_exporter_folds"}
    for label, genome in genomes:
        model, info = build_model(genome, seed=0)
        if label.startswith("op:"):
            assert info.shortcuts == ("projection", "none", "projection")
        assert_matches_plan(fold_batchnorm(model), info, genome, batch=1)
