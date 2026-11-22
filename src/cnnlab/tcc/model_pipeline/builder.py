"""PyTorch network ``A_g`` built from an expressed genome v1 (TCC-02 sections 2 and 3).

The builder turns :class:`~cnnlab.tcc.science.genome.ExpressedGenome` into
the trainable FP32 network *before* any representation transform. It follows
the topology of :func:`~cnnlab.tcc.science.expression.plan_network`:

* stem: 3x3 conv (3 -> ``stem_channels``) + BN + activation, resolution 32;
* each block reduces at its start (``maxpool``/``avgpool`` as the first op of
  the body, ``conv_s2`` as stride 2 in the first spatial convolution);
* MBConv expansion is taken over the unit input (``C_in * e`` for the first
  unit, ``C_out * e`` for the next ones);
* ``residual=True`` adds one shortcut around the whole block: identity when
  width and resolution are kept, otherwise a 1x1 conv (stride 2 if the block
  reduces) + BN projection, which is part of the model and of its cost;
* head: global average pooling, optional FC256 + activation, dropout with
  ``head_dropout`` and a linear layer with 100 logits.

``n_convs`` counts repeated units of the block op: conv-BN-act for
``standard``/``grouped_g4``, depthwise+pointwise pairs for ``dw_separable``
and inverted-residual units (expand 1x1, depthwise kxk, linear project 1x1)
for MBConv. MBConv blocks have a linear output (no activation after the add),
as in MobileNetV2; the other ops activate after the add.

Not implemented here, and reported by :class:`BuildInfo` instead of being
silently approximated: physical pruning (``prune_pct``), per-region precision
and INT8 Q/DQ (``precision``, ``stem_precision``, ``head_precision`` and the
INT8 genes). They belong to the post-training representation step. Training
genes (``log10_lr``, ``log10_wd``) configure the optimizer, not the module.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from cnnlab.tcc.science.expression import BlockPlan, NetworkPlan, plan_network
from cnnlab.tcc.science.genome import ExpressedGenome
from cnnlab.tcc.science.space import (
    GROUPED_CONV_GROUPS,
    HEAD_HIDDEN_UNITS,
    INPUT_RESOLUTION,
    MAX_SPATIAL_REDUCTIONS,
    MBCONV_EXPANSION,
    NUM_CLASSES,
)

BUILDER_VERSION = "tcc-builder/v1"
INPUT_CHANNELS = 3

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "hardswish": nn.Hardswish,
}

# Genes that shape the trained module; everything else is reported as deferred.
_REPRESENTATION_GENES = ("stem_precision", "head_precision", "int8_mode", "weight_granularity", "calibration")
_TRAINING_GENES = ("log10_lr", "log10_wd")


class ModelBuildError(ValueError):
    """The expressed genome has no buildable topology (candidate-level invalidity).

    ``code`` is stable so the evaluator can map it to ``failure_code``.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """What the builder materialized and which expressed genes it left to later steps."""

    builder_version: str
    seed: int
    block_out_shapes: tuple[tuple[int, int, int], ...]  # (channels, height, width) per block
    shortcuts: tuple[str, ...]
    parameters: int
    deferred_representation: tuple[str, ...]
    deferred_training: tuple[str, ...]
    unpruned: bool = True  # the builder never prunes: the evaluator prunes the trained network (pruning.py)


def conv_bn(cin: int, cout: int, kernel: int, stride: int = 1, groups: int = 1) -> list[nn.Module]:
    """Bias-free convolution followed by BN, with 'same' padding for odd kernels."""
    return [
        nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    ]


class Block(nn.Module):
    """One active block: optional reduction, ``n_convs`` units of ``op`` and a shortcut."""

    def __init__(self, plan: BlockPlan, kernel: int, n_convs: int, downsample: str, activation: str) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        self.op = plan.op
        self.linear_output = plan.op in MBCONV_EXPANSION
        reduces = downsample != "none"
        layers: list[nn.Module] = []
        if downsample == "maxpool":
            layers.append(nn.MaxPool2d(2))
        elif downsample == "avgpool":
            layers.append(nn.AvgPool2d(2))
        first_stride = 2 if downsample == "conv_s2" else 1

        cin, cout = plan.in_channels, plan.out_channels
        for unit in range(n_convs):
            stride = first_stride if unit == 0 else 1
            last = unit == n_convs - 1
            # Units that end the residual branch leave the activation to after the add.
            end_act = not (last and plan.shortcut != "none")
            if plan.op in ("standard", "grouped_g4"):
                groups = GROUPED_CONV_GROUPS if plan.op == "grouped_g4" else 1
                layers += conv_bn(cin, cout, kernel, stride, groups)
                if end_act:
                    layers.append(act())
            elif plan.op == "dw_separable":
                layers += conv_bn(cin, cin, kernel, stride, groups=cin)
                layers.append(act())
                layers += conv_bn(cin, cout, 1)
                if end_act:
                    layers.append(act())
            else:  # MBConv: expand, depthwise, linear projection
                hidden = cin * MBCONV_EXPANSION[plan.op]
                layers += conv_bn(cin, hidden, 1)
                layers.append(act())
                layers += conv_bn(hidden, hidden, kernel, stride, groups=hidden)
                layers.append(act())
                layers += conv_bn(hidden, cout, 1)
            cin = cout
        self.body = nn.Sequential(*layers)

        if plan.shortcut == "projection":
            self.shortcut: nn.Module | None = nn.Sequential(*conv_bn(plan.in_channels, cout, 1, 2 if reduces else 1))
        elif plan.shortcut == "identity":
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None
        self.post_act: nn.Module = act() if self.shortcut is not None and not self.linear_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.shortcut is not None:
            y = self.post_act(y + self.shortcut(x))
        return y


class Head(nn.Module):
    """GAP, optional FC256 + activation, dropout and the 100-logit classifier."""

    def __init__(self, in_channels: int, hidden: int | None, dropout: float, activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]
        if hidden is not None:
            layers += [nn.Linear(in_channels, hidden), ACTIVATIONS[activation]()]
            in_channels = hidden
        layers += [nn.Dropout(dropout), nn.Linear(in_channels, NUM_CLASSES)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TCCNet(nn.Module):
    """``stem -> blocks -> head``; regions are separate modules so a later step can type them."""

    def __init__(self, stem: nn.Module, blocks: list[Block], head: Head) -> None:
        super().__init__()
        self.stem = stem
        self.blocks = nn.ModuleList(blocks)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def check_buildable(plan: NetworkPlan) -> None:
    """Raise :class:`ModelBuildError` for topologies the builder cannot realize.

    With v1 domains ``express`` already guarantees these; the builder checks
    again because it may receive a genome deserialized from storage.
    """
    for b in plan.blocks:
        if b.op == "grouped_g4" and (b.in_channels % GROUPED_CONV_GROUPS or b.out_channels % GROUPED_CONV_GROUPS):
            raise ModelBuildError(
                "R2_grouped_divisibility", f"block {b.index}: {b.in_channels}->{b.out_channels} not divisible by 4"
            )
        if b.out_resolution < INPUT_RESOLUTION >> MAX_SPATIAL_REDUCTIONS:
            raise ModelBuildError("R1_resolution", f"block {b.index}: resolution {b.out_resolution} below 2")


def build_model(expressed: ExpressedGenome, seed: int) -> tuple[TCCNet, BuildInfo]:
    """Build the FP32 training network for ``expressed`` with weights drawn from ``seed``.

    The global torch RNG is left untouched, so building a model never shifts
    the data-order or dropout streams of the caller.
    """
    if not isinstance(expressed, ExpressedGenome):
        raise TypeError(f"build_model needs an ExpressedGenome, got {type(expressed).__name__}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")
    plan = plan_network(expressed)
    check_buildable(plan)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        stem = nn.Sequential(*conv_bn(INPUT_CHANNELS, expressed.stem_channels, 3), ACTIVATIONS[expressed.activation]())
        blocks = [
            Block(p, genes.kernel, genes.n_convs, genes.downsample, expressed.activation)
            for p, genes in zip(plan.blocks, expressed.blocks)
        ]
        hidden = HEAD_HIDDEN_UNITS if expressed.head == "gap_fc256_linear" else None
        head = Head(plan.blocks[-1].out_channels, hidden, expressed.head_dropout, expressed.activation)
        model = TCCNet(stem, blocks, head)
    deferred = [g for g in _REPRESENTATION_GENES if getattr(expressed, g) is not None]
    deferred += [f"blocks[{i}].precision" for i in range(expressed.n_blocks)]
    deferred += [f"blocks[{i}].prune_pct" for i, b in enumerate(expressed.blocks) if b.prune_pct]
    info = BuildInfo(
        builder_version=BUILDER_VERSION,
        seed=seed,
        block_out_shapes=tuple((p.out_channels, p.out_resolution, p.out_resolution) for p in plan.blocks),
        shortcuts=tuple(p.shortcut for p in plan.blocks),
        parameters=count_parameters(model),
        deferred_representation=tuple(deferred),
        deferred_training=_TRAINING_GENES,
    )
    return model, info


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Eval-mode copy with every ``Conv2d -> BatchNorm2d`` pair fused into the conv.

    Folding in PyTorch keeps the fused weights in the conv's dtype. In wave 2A
    the dynamo exporter folded BN itself and wrote FLOAT weights next to a
    BF16 activation, which TensorRT 11.3 rejected when
    tested on the GPU).
    """
    folded = copy.deepcopy(model).eval()
    for module in folded.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for (name_a, a), (name_b, b) in zip(children, children[1:]):
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                setattr(module, name_a, fuse_conv_bn_eval(a, b))
                setattr(module, name_b, nn.Identity())
    return folded


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters (auxiliary metric; BN running statistics excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def effective_widths(model: TCCNet, resolution: int = INPUT_RESOLUTION) -> dict[str, Any]:
    """Channels the network really has, read off a forward pass (so, after any pruning).

    ``stem`` and ``blocks`` are the output channels of the stem and of each block; ``head_hidden`` is the width of
    the head's hidden layer (None for the plain linear head).
    """
    seen: dict[str, int] = {}
    handles = [model.stem.register_forward_hook(lambda _m, _i, out: seen.__setitem__("stem", out.shape[1]))]
    for index, block in enumerate(model.blocks):
        handles.append(
            block.register_forward_hook(lambda _m, _i, out, key=f"blocks.{index}": seen.__setitem__(key, out.shape[1]))
        )
    first = next(model.parameters())
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, INPUT_CHANNELS, resolution, resolution, device=first.device, dtype=first.dtype))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    return {
        "stem": seen["stem"],
        "blocks": [seen[f"blocks.{i}"] for i in range(len(model.blocks))],
        "head_hidden": linears[0].out_features if len(linears) > 1 else None,
    }


def count_macs(model: nn.Module, resolution: int = INPUT_RESOLUTION) -> int:
    """Multiply-accumulates of convolutions and linear layers for one image (auxiliary for H4).

    Pooling, BN, activations and additions are not counted. Runs one eval-mode
    forward on zeros, on the model's current device and dtype.
    """
    total = 0

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        kh, kw = module.kernel_size
        total += output.numel() * (module.in_channels // module.groups) * kh * kw

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    first = next(model.parameters())
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, INPUT_CHANNELS, resolution, resolution, device=first.device, dtype=first.dtype))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    return total
