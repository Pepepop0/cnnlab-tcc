"""Physical structured pruning of a trained :class:`TCCNet` (TCC-02 section 3, rule 3).

Channels are removed for real: convolutions, BatchNorm layers and the head get
smaller tensors, so the engine compiled afterwards is smaller and its cost is
measured, not estimated. Pruning runs after training and before the fine-tune
epoch, the quantization and the export.

Channel groups
    Every tensor whose channels must be removed together is one *group*. A group
    has producers (the convolutions that write it, with their BN), depthwise
    members (depthwise convolutions whose input and output are the group itself)
    and consumers (convolutions and the first head layer that read it). The
    groups follow from the architecture of :mod:`builder`:

    * the stem output, never pruned (the genome has no gene for it);
    * per block, every intermediate unit output and every MBConv hidden tensor,
      pruned by the block's ``prune_pct``;
    * the block output, pruned by the block's ``prune_pct``, which a projection
      shortcut also writes.

Identity shortcuts (decision of 05/10/2026)
    ``y = body(x) + x`` forces the block output to keep exactly the channels of
    its input. The width of such a tensor is decided by the block that created
    it; a block with an identity shortcut does not prune its output (it follows
    its input) but prunes its internal channels with its own ``prune_pct``. The
    requested and the realized fraction of every group are reported, so a gene
    that had no effect is visible, never hidden.

Selection and rounding
    Kept channels: :func:`~cnnlab.tcc.science.expression.pruned_width`, i.e. the
    requested fraction rounded half-up to a multiple of 16, at least 16.
    Importance of a channel: L1 norm of the filters that produce it, summed over
    the group's producers. A group touched by a grouped convolution (``g = 4``)
    keeps the same number of channels in each of the four contiguous chunks, so
    the group structure survives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from cnnlab.tcc.model_pipeline.builder import TCCNet, count_parameters
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.expression import pruned_width

PRUNING_VERSION = "tcc-pruning/v1"
IMPORTANCE = "l1-of-producing-filters"


def finetune_config(train: TrainConfig) -> TrainConfig:
    """The single fine-tune epoch after pruning (TCC-02 section 5): no warm-up, cosine from the gene's lr.

    The weights are already trained, so warming up from zero would only waste the epoch.
    """
    return replace(train, epochs=1, warmup_epochs=0)


class PruningError(ValueError):
    """The network cannot be pruned as requested without breaking its topology (invalid candidate)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class ChannelGroup:
    """Channels that must be removed together, with every layer that touches them."""

    name: str
    size: int
    requested_pct: int
    producers: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = field(default_factory=list)
    depthwise: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = field(default_factory=list)
    consumers: list[nn.Conv2d] = field(default_factory=list)
    linear_consumers: list[nn.Linear] = field(default_factory=list)
    kept: tuple[int, ...] = ()

    @property
    def balanced(self) -> bool:
        """Touched by a grouped (non-depthwise) convolution: keep equal counts per group chunk."""
        convs = [c for c, _ in self.producers] + self.consumers
        return any(c.groups > 1 and not _is_depthwise(c) for c in convs)

    @property
    def chunks(self) -> int:
        convs = [c for c, _ in self.producers] + self.consumers
        groups = {c.groups for c in convs if c.groups > 1 and not _is_depthwise(c)}
        if len(groups) > 1:
            raise PruningError("mixed_groups", f"{self.name}: grouped convolutions disagree ({sorted(groups)})")
        return groups.pop() if groups else 1

    def summary(self) -> dict[str, Any]:
        kept = len(self.kept) if self.kept else self.size
        return {
            "group": self.name,
            "channels": self.size,
            "kept": kept,
            "requested_pct": self.requested_pct,
            "realized_pct": round(100 * (1 - kept / self.size), 2),
            "balanced": self.balanced,
        }


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What was removed; stored with the attempt (requested and realized per group)."""

    version: str
    importance: str
    groups: tuple[dict[str, Any], ...]
    blocks: tuple[dict[str, Any], ...]
    parameters_before: int
    parameters_after: int

    @property
    def pruned(self) -> bool:
        return self.parameters_after < self.parameters_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "importance": self.importance,
            "groups": list(self.groups),
            "blocks": list(self.blocks),
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
        }


def _is_depthwise(conv: nn.Conv2d) -> bool:
    return conv.groups > 1 and conv.groups == conv.in_channels == conv.out_channels


def _conv_bn_pairs(body: nn.Sequential) -> list[tuple[nn.Conv2d, nn.BatchNorm2d]]:
    layers = list(body)
    pairs = []
    for a, b in zip(layers, layers[1:]):
        if isinstance(a, nn.Conv2d):
            if not isinstance(b, nn.BatchNorm2d):
                raise PruningError("unexpected_layout", f"convolution not followed by BatchNorm: {a}")
            pairs.append((a, b))
    if isinstance(layers[-1], nn.Conv2d):
        raise PruningError("unexpected_layout", "block body ends with a convolution without BatchNorm")
    return pairs


def channel_groups(model: TCCNet, prune_pcts: Sequence[int]) -> list[ChannelGroup]:
    """Groups of ``model`` (in forward order) with the requested fraction of each.

    ``prune_pcts`` has one entry per active block, in order.
    """
    if len(prune_pcts) != len(model.blocks):
        raise PruningError("bad_request", f"{len(prune_pcts)} prune_pct values for {len(model.blocks)} blocks")
    groups: list[ChannelGroup] = []

    def new(name: str, size: int, pct: int) -> ChannelGroup:
        group = ChannelGroup(name, size, pct)
        groups.append(group)
        return group

    stem_conv, stem_bn = model.stem[0], model.stem[1]
    block_in = new("stem", stem_conv.out_channels, 0)
    block_in.producers.append((stem_conv, stem_bn))
    for index, (block, pct) in enumerate(zip(model.blocks, prune_pcts)):
        identity = isinstance(block.shortcut, nn.Identity)
        pairs = _conv_bn_pairs(block.body)
        free = [i for i, (conv, _) in enumerate(pairs) if not _is_depthwise(conv)]
        if not free:
            raise PruningError("unexpected_layout", f"block {index} has no channel-mixing convolution")
        current = block_in
        inner = 0
        for i, (conv, bn) in enumerate(pairs):
            if _is_depthwise(conv):
                current.depthwise.append((conv, bn))
                continue
            current.consumers.append(conv)
            if i == free[-1]:
                out = block_in if identity else new(f"block{index}.out", conv.out_channels, pct)
            else:
                out = new(f"block{index}.inner{inner}", conv.out_channels, pct)
                inner += 1
            out.producers.append((conv, bn))
            current = out
        if isinstance(block.shortcut, nn.Sequential):
            proj, proj_bn = block.shortcut[0], block.shortcut[1]
            block_in.consumers.append(proj)
            current.producers.append((proj, proj_bn))
        block_in = current
    first_linear = next((m for m in model.head.modules() if isinstance(m, nn.Linear)), None)
    if first_linear is None:
        raise PruningError("unexpected_layout", "head has no linear layer")
    block_in.linear_consumers.append(first_linear)
    for group in groups:
        sizes = {c.out_channels for c, _ in group.producers} | {c.in_channels for c in group.consumers}
        sizes |= {c.in_channels for c, _ in group.depthwise} | {lin.in_features for lin in group.linear_consumers}
        if sizes != {group.size}:
            raise PruningError("inconsistent_widths", f"{group.name}: layers disagree on width {sorted(sizes)}")
    return groups


def select_channels(group: ChannelGroup) -> tuple[int, ...]:
    """Indices kept by ``group``, sorted; all of them when nothing is requested."""
    keep = group.size if group.requested_pct == 0 else pruned_width(group.size, group.requested_pct)
    if keep == group.size:
        return tuple(range(group.size))
    importance = torch.zeros(group.size)
    for conv, _ in group.producers:
        importance += conv.weight.detach().abs().flatten(1).sum(1).float().cpu()
    chunks = group.chunks
    if keep % chunks:
        raise PruningError("unbalanced_groups", f"{group.name}: {keep} channels do not split into {chunks} groups")
    size, per = group.size // chunks, keep // chunks
    kept: list[int] = []
    for c in range(chunks):
        scores = importance[c * size : (c + 1) * size]
        # Highest importance first; ties go to the lower index, so the choice is deterministic.
        order = sorted(range(size), key=lambda i: (-float(scores[i]), i))[:per]
        kept += [c * size + i for i in sorted(order)]
    return tuple(kept)


def _param(tensor: torch.Tensor, like: nn.Parameter) -> nn.Parameter:
    return nn.Parameter(tensor.detach().clone().contiguous(), requires_grad=like.requires_grad)


def _prune_bn(bn: nn.BatchNorm2d, idx: torch.Tensor) -> None:
    bn.weight = _param(bn.weight[idx], bn.weight)
    bn.bias = _param(bn.bias[idx], bn.bias)
    bn.running_mean = bn.running_mean[idx].clone()
    bn.running_var = bn.running_var[idx].clone()
    bn.num_features = int(idx.numel())


def _prune_out(conv: nn.Conv2d, idx: torch.Tensor) -> None:
    conv.weight = _param(conv.weight[idx], conv.weight)
    if conv.bias is not None:
        conv.bias = _param(conv.bias[idx], conv.bias)
    conv.out_channels = int(idx.numel())


def _prune_in(conv: nn.Conv2d, kept: Sequence[int]) -> None:
    if conv.groups == 1:
        idx = torch.tensor(kept, device=conv.weight.device)
        conv.weight = _param(conv.weight[:, idx], conv.weight)
        conv.in_channels = len(kept)
        return
    # Grouped: output group g reads input chunk g; keep that chunk's surviving positions.
    groups = conv.groups
    in_per, out_per = conv.in_channels // groups, conv.out_channels // groups
    rows = []
    for g in range(groups):
        positions = [i - g * in_per for i in kept if g * in_per <= i < (g + 1) * in_per]
        pos = torch.tensor(positions, device=conv.weight.device)
        rows.append(conv.weight[g * out_per : (g + 1) * out_per][:, pos])
    conv.weight = _param(torch.cat(rows), conv.weight)
    conv.in_channels = len(kept)


def _apply(group: ChannelGroup) -> None:
    if len(group.kept) == group.size:
        return
    for conv, bn in group.producers:
        idx = torch.tensor(group.kept, device=conv.weight.device)
        _prune_out(conv, idx)
        _prune_bn(bn, idx)
    for conv, bn in group.depthwise:
        idx = torch.tensor(group.kept, device=conv.weight.device)
        _prune_out(conv, idx)
        conv.in_channels = conv.groups = int(idx.numel())
        _prune_bn(bn, idx)
    for conv in group.consumers:
        _prune_in(conv, group.kept)
    for linear in group.linear_consumers:
        idx = torch.tensor(group.kept, device=linear.weight.device)
        linear.weight = _param(linear.weight[:, idx], linear.weight)
        linear.in_features = len(group.kept)


def prune_network(model: TCCNet, prune_pcts: Sequence[int]) -> PruneReport:
    """Remove channels from ``model`` in place, as requested per active block.

    Selection uses the trained weights; layers shared by several groups (a
    convolution that reads one group and writes another) are cut once per side.
    """
    before = count_parameters(model)
    groups = channel_groups(model, prune_pcts)
    for group in groups:
        group.kept = select_channels(group)
    with torch.no_grad():
        for group in groups:
            _apply(group)
    summaries = tuple(g.summary() for g in groups)
    blocks = []
    for index, pct in enumerate(prune_pcts):
        own = [s for s in summaries if s["group"].startswith(f"block{index}.")]
        blocks.append(
            {
                "block": index,
                "requested_pct": pct,
                "output_follows_input": isinstance(model.blocks[index].shortcut, nn.Identity),
                "realized_pct": {s["group"].split(".", 1)[1]: s["realized_pct"] for s in own},
            }
        )
    return PruneReport(PRUNING_VERSION, IMPORTANCE, summaries, tuple(blocks), before, count_parameters(model))
