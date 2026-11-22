"""Physical structured pruning (CPU): the pruned network computes what the masked original computes.

Oracle: in a copy of the original network, every weight that *reads* a removed
channel is set to zero. Removed channels then contribute nothing downstream, so
that copy and the physically pruned network must give the same logits. This
checks the surgery for every op, shortcut kind and grouped convolution, with
weights drawn by the builder (no training needed).
"""

from __future__ import annotations

import copy
import random

import pytest
import torch
from torch import nn

from cnnlab.tcc.model_pipeline.builder import TCCNet, build_model
from cnnlab.tcc.model_pipeline.pruning import (
    PruningError,
    channel_groups,
    prune_network,
    select_channels,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1

SEEDS = range(40)


def network(seed: int) -> TCCNet:
    genome = express(sample_genome(SPACE_V1, GeneRng(seed)), SPACE_V1).genome
    model, _ = build_model(genome, seed)
    return model.eval()


def requests(model: TCCNet, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.choice((0, 25, 50)) for _ in model.blocks]


def masked(original: TCCNet, pcts: list[int]) -> TCCNet:
    """Copy of ``original`` where every reader of a channel that pruning removes gets zero weights."""
    copy_ = copy.deepcopy(original)
    groups = channel_groups(copy_, pcts)
    selections = [set(select_channels(g)) for g in groups]  # all chosen before any weight is touched
    with torch.no_grad():
        for group, kept in zip(groups, selections):
            removed = [i for i in range(group.size) if i not in kept]
            if not removed:
                continue
            for conv in group.consumers:
                if conv.groups == 1:
                    conv.weight[:, removed] = 0
                    continue
                in_per, out_per = conv.in_channels // conv.groups, conv.out_channels // conv.groups
                for i in removed:
                    g = i // in_per
                    conv.weight[g * out_per : (g + 1) * out_per, i - g * in_per] = 0
            for linear in group.linear_consumers:
                linear.weight[:, removed] = 0
    return copy_


@pytest.mark.parametrize("seed", SEEDS)
def test_pruned_network_matches_the_masked_original(seed: int) -> None:
    original = network(seed)
    pcts = requests(original, seed)
    oracle = masked(original, pcts)
    pruned = copy.deepcopy(original)
    report = prune_network(pruned, pcts)
    x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        assert torch.allclose(pruned.eval()(x), oracle.eval()(x), atol=1e-4, rtol=1e-4)
    assert report.parameters_after <= report.parameters_before
    for conv in (m for m in pruned.modules() if isinstance(m, nn.Conv2d)):
        assert conv.weight.shape[0] == conv.out_channels
        assert conv.weight.shape[1] == conv.in_channels // conv.groups
        assert conv.out_channels % 16 == 0 and (conv.in_channels == 3 or conv.in_channels % 16 == 0)


def test_no_request_changes_nothing() -> None:
    original = network(7)
    pruned = copy.deepcopy(original)
    report = prune_network(pruned, [0] * len(original.blocks))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(pruned(x), original(x))
    assert report.parameters_after == report.parameters_before and not report.pruned
    assert all(g["realized_pct"] == 0 for g in report.groups)


def test_grouped_convolutions_keep_equal_counts_per_group() -> None:
    seen = 0
    for seed in SEEDS:
        model = network(seed)
        pcts = [50] * len(model.blocks)
        for group in channel_groups(model, pcts):
            if not group.balanced:
                continue
            kept = select_channels(group)
            chunk = group.size // group.chunks
            counts = [sum(1 for i in kept if c * chunk <= i < (c + 1) * chunk) for c in range(group.chunks)]
            assert len(set(counts)) == 1
            seen += 1
    assert seen, "no grouped convolution in the sampled networks"


def test_identity_block_output_follows_its_input() -> None:
    for seed in range(200):
        model = network(seed)
        identity = [i for i, b in enumerate(model.blocks) if isinstance(b.shortcut, nn.Identity)]
        if identity:
            break
    else:
        pytest.fail("no identity shortcut in the sampled networks")
    index = identity[0]
    pcts = [0] * len(model.blocks)
    pcts[index] = 50
    report = prune_network(model, pcts)
    assert report.blocks[index]["output_follows_input"] is True
    assert "out" not in report.blocks[index]["realized_pct"]  # the output is not a group of its own
    with torch.no_grad():
        y = model.stem(torch.randn(2, 3, 32, 32))
        for block in model.blocks[:index]:
            y = block(y)
        assert model.blocks[index](y).shape == y.shape


def test_rounding_reports_requested_and_realized() -> None:
    model = network(3)
    pcts = [25] * len(model.blocks)
    report = prune_network(model, pcts)
    for group in report.groups:
        if group["group"] == "stem":
            assert group["requested_pct"] == 0 and group["realized_pct"] == 0
            continue
        assert group["kept"] % 16 == 0 and group["kept"] >= 16
        assert group["requested_pct"] == 25


def test_selection_is_deterministic() -> None:
    a, b = network(11), network(11)
    ra, rb = prune_network(a, [50] * len(a.blocks)), prune_network(b, [50] * len(b.blocks))
    assert ra.groups == rb.groups
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_pruned_network_still_trains() -> None:
    model = network(5).train()
    prune_network(model, [50] * len(model.blocks))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = nn.functional.cross_entropy(model(torch.randn(8, 3, 32, 32)), torch.randint(0, 100, (8,)))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_wrong_number_of_requests_is_refused() -> None:
    model = network(1)
    with pytest.raises(PruningError, match="bad_request"):
        channel_groups(model, [0] * (len(model.blocks) + 1))


def test_finetune_is_one_epoch_without_warmup() -> None:
    from cnnlab.tcc.model_pipeline.pruning import finetune_config
    from cnnlab.tcc.model_pipeline.trainer import TrainConfig, lr_factor

    config = finetune_config(TrainConfig(epochs=5, batch_size=128))
    assert (config.epochs, config.warmup_epochs, config.batch_size) == (1, 0, 128)
    assert lr_factor(0, 0, 100) == 1.0 and lr_factor(99, 0, 100) < 0.01
