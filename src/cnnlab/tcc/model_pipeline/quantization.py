"""INT8 post-training quantization with explicit Q/DQ (task G3).

TCC-02 rule 4: INT8 is symmetric, quantization is explicit (QuantizeLinear /
DequantizeLinear nodes in the ONNX graph), and the activation ranges are
calibrated only on the 1,024 frozen training examples, never on validation or
test data. An INT8 region quantizes the input and the weights of every
convolution and linear layer inside it; the activations between layers stay in
FP32 and the engine fuses the Q/DQ pairs into INT8 kernels. Whether it really
did is read from the engine inspector (see ``precision.check_engine_precision``).

Calibration methods, applied to the input of each quantized layer over the
calibration set (histogram of absolute values, 2,048 bins):

* ``max``: the largest absolute value;
* ``percentile_99_99``: the 99.99th percentile of the absolute values;
* ``entropy``: the clipping threshold that minimizes the KL divergence between
  the FP32 histogram and its 128-level quantization (TensorRT's method).

Weights use the maximum absolute value, per output channel (``per_channel``) or
of the whole tensor (``per_tensor``). Everything is deterministic: the same
network, calibration images and method give the same scales.

QAT (task G4) starts from the same Q/DQ network, with its activation ranges
calibrated by ``QAT_INIT_CALIBRATION`` (the expressed genome has no calibration
gene in QAT mode), and fine-tunes it for one epoch with the quantization
simulated in the forward pass. Rounding is passed through unchanged in the
backward pass (straight-through estimator). While training, the activation
ranges follow an exponential moving average of the batch maxima and the weight
scales follow the weights; the last values are the ones exported.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QUANTIZATION_POLICY = "tcc-quantization/int8-symmetric-explicit-qdq-v1"
N_CALIBRATION = 1024
INT8_MAX = 127
QMIN, QMAX = -128, 127
HISTOGRAM_BINS = 2048
ENTROPY_LEVELS = 128
ENTROPY_STRIDE = 4  # candidate thresholds every 4 bins: same optimum within one bin width, 4x faster
PERCENTILE = 99.99
QAT_INIT_CALIBRATION = "percentile_99_99"  # robust to outliers, and cheaper than the entropy search
QAT_RANGE_MOMENTUM = 0.9  # weight of the previous activation range in each training step
MODES: tuple[str, ...] = ("ptq", "qat")
CALIBRATIONS: tuple[str, ...] = ("max", "entropy", "percentile_99_99")
GRANULARITIES: tuple[str, ...] = ("per_tensor", "per_channel")
MIN_RANGE = 1e-8


class QuantizationError(ValueError):
    """The network cannot be quantized as requested (failure_stage ``quantize``)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# --- what is calibrated -------------------------------------------------------------------------------------


def calibration_indices(
    labels: Sequence[int] | np.ndarray | torch.Tensor, n: int = N_CALIBRATION, seed: int = 0
) -> list[int]:
    """Positions of ``n`` calibration examples inside the training labels, class-balanced and seeded.

    Positions refer to the training array that is passed in, so the calibration
    set is a subset of the training data by construction. The same labels and
    seed always give the same positions.
    """
    y = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels, dtype=np.int64)
    if n > len(y):
        raise QuantizationError("calibration_too_small", f"{n} calibration examples requested from {len(y)}")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = [list(rng.permutation(np.flatnonzero(y == c))) for c in classes]
    chosen: list[int] = []
    while len(chosen) < n:  # round robin over classes: balanced to within one example
        for pool in per_class:
            if pool and len(chosen) < n:
                chosen.append(int(pool.pop()))
    return sorted(chosen)


def indices_sha256(indices: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(indices), dtype=np.int64).tobytes()).hexdigest()


def quantizable_layers(net: nn.Module, regions: Iterable[str]) -> dict[str, nn.Module]:
    """Convolution and linear layers of ``net`` whose name starts with one of ``regions`` (``stem``, ``blocks.2``...)."""
    prefixes = tuple(f"{r}." for r in regions)
    return {
        name: module
        for name, module in net.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name.startswith(prefixes)
    }


# --- calibration ------------------------------------------------------------------------------------------


def entropy_threshold(
    hist: np.ndarray, bin_width: float, levels: int = ENTROPY_LEVELS, stride: int = ENTROPY_STRIDE
) -> float:
    """Clipping threshold with the smallest KL divergence between the histogram and its ``levels``-level version."""
    h = np.asarray(hist, dtype=np.float64).copy()
    n = len(h)
    if h.sum() <= 0:
        return MIN_RANGE
    h[0] = h[1] if n > 1 else h[0]  # the zero bin is dominated by exact zeros (ReLU), not by signal
    best_i, best_kl = n, np.inf
    for i in range(levels, n + 1, stride):
        bounds = (np.arange(levels + 1) * i // levels).astype(np.int64)
        window = h[:i]
        ref = window.copy()
        ref[-1] += h[i:].sum()  # clipped outliers land on the last kept bin
        nonzero = (window > 0).astype(np.float64)
        mass = np.add.reduceat(window, bounds[:-1])
        count = np.add.reduceat(nonzero, bounds[:-1])
        spread = np.repeat(mass / np.maximum(count, 1.0), np.diff(bounds)) * nonzero
        p, q = ref / ref.sum(), spread / max(spread.sum(), MIN_RANGE)
        mask = p > 0
        if np.any(mask & (q <= 0)):  # tail mass on an empty bin: smooth instead of an infinite divergence
            q = np.where(q > 0, q, 1e-12)
        kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        if kl < best_kl:
            best_kl, best_i = kl, i
    return max((best_i) * bin_width, MIN_RANGE)


def percentile_threshold(hist: np.ndarray, bin_width: float, percentile: float = PERCENTILE) -> float:
    total = float(np.sum(hist))
    if total <= 0:
        return MIN_RANGE
    cdf = np.cumsum(hist, dtype=np.float64) / total
    index = int(np.searchsorted(cdf, percentile / 100.0, side="left"))
    return max((min(index, len(hist) - 1) + 1) * bin_width, MIN_RANGE)


def threshold_from_histogram(hist: np.ndarray, amax: float, method: str) -> float:
    """Activation range chosen by ``method`` from a histogram of absolute values over ``[0, amax]``."""
    if method == "max":
        return max(float(amax), MIN_RANGE)
    width = max(float(amax), MIN_RANGE) / len(hist)
    if method == "percentile_99_99":
        return min(percentile_threshold(hist, width), max(float(amax), MIN_RANGE))
    if method == "entropy":
        return min(entropy_threshold(hist, width), max(float(amax), MIN_RANGE))
    raise QuantizationError("unknown_calibration", f"{method!r} is not one of {CALIBRATIONS}")


@torch.no_grad()
def calibrate(
    net: nn.Module, layers: Mapping[str, nn.Module], images: torch.Tensor, method: str, batch_size: int = 256
) -> dict[str, float]:
    """Activation range at the input of every layer in ``layers``, from forward passes over ``images``.

    ``net`` is the whole FP32 network (the layers are modules inside it); it is
    run in eval mode. Two passes: the first finds each layer's largest value,
    the second fills the histograms (skipped for ``max``).
    """
    if method not in CALIBRATIONS:
        raise QuantizationError("unknown_calibration", f"{method!r} is not one of {CALIBRATIONS}")
    if not layers:
        return {}
    was_training = net.training
    net.eval()
    amax = dict.fromkeys(layers, 0.0)
    hists: dict[str, torch.Tensor] = {}

    def run(hook_for: Callable[[str], Callable[..., None]]) -> None:
        handles = [m.register_forward_pre_hook(hook_for(name)) for name, m in layers.items()]
        try:
            for start in range(0, len(images), batch_size):
                net(images[start : start + batch_size])
        finally:
            for handle in handles:
                handle.remove()

    def track_max(name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            amax[name] = max(amax[name], float(args[0].detach().abs().max()))

        return hook

    def fill(name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            top = max(amax[name], MIN_RANGE)
            counts = torch.histc(args[0].detach().abs().float().flatten(), bins=HISTOGRAM_BINS, min=0.0, max=top)
            hists[name] = hists[name] + counts.double() if name in hists else counts.double()

        return hook

    run(track_max)
    if method != "max":
        run(fill)
    net.train(was_training)
    return {
        name: threshold_from_histogram(hists[name].cpu().numpy() if name in hists else np.zeros(1), amax[name], method)
        for name in layers
    }


# --- Q/DQ layers ------------------------------------------------------------------------------------------


def _scale_buffer(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().clamp_min(MIN_RANGE) / INT8_MAX


class QDQLayer(nn.Module):
    """A convolution or linear layer with Q/DQ on its input and on its weights (symmetric INT8).

    ``fake_quantize_*`` exports to QuantizeLinear/DequantizeLinear, which is the
    explicit quantization TensorRT reads in strongly typed mode. The scales are
    buffers: PTQ calibrates them once; with ``qat`` they are updated at every
    training step (see the module docstring), so the exported values are the
    last ones.
    """

    def __init__(self, layer: nn.Conv2d | nn.Linear, input_range: float, granularity: str, qat: bool = False) -> None:
        super().__init__()
        if granularity not in GRANULARITIES:
            raise QuantizationError("unknown_granularity", f"{granularity!r} is not one of {GRANULARITIES}")
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            raise QuantizationError("unsupported_layer", f"cannot quantize {type(layer).__name__}")
        self.layer = layer
        self.granularity = granularity
        self.qat = qat
        device = layer.weight.device
        self.register_buffer("input_amax", torch.tensor(max(float(input_range), MIN_RANGE), device=device))
        self.register_buffer("input_scale", _scale_buffer(torch.tensor(float(input_range), device=device)))
        self.register_buffer("input_zero", torch.zeros((), dtype=torch.int32, device=device))
        self.refresh_weight_scale()

    @property
    def is_conv(self) -> bool:
        return isinstance(self.layer, nn.Conv2d)

    def _weight_scale(self) -> torch.Tensor:
        weight = self.layer.weight.detach()
        if self.granularity == "per_channel":
            return _scale_buffer(weight.abs().flatten(1).amax(dim=1))
        return _scale_buffer(weight.abs().amax())

    def refresh_weight_scale(self) -> None:
        """Weight scale from the current weights (max, per output channel or per tensor)."""
        scale = self._weight_scale()
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_zero", torch.zeros(scale.shape, dtype=torch.int32, device=scale.device))

    @torch.no_grad()
    def _follow(self, x: torch.Tensor) -> None:
        """QAT step: activation range by moving average of the batch maximum, weight scale from the weights."""
        batch_max = x.detach().abs().max().float().clamp_min(MIN_RANGE)
        self.input_amax.mul_(QAT_RANGE_MOMENTUM).add_((1.0 - QAT_RANGE_MOMENTUM) * batch_max)
        self.input_scale.copy_(self.input_amax / INT8_MAX)
        self.weight_scale.copy_(self._weight_scale())

    def quantized_weight(self) -> torch.Tensor:
        weight = self.layer.weight
        if self.granularity == "per_channel":
            return torch.fake_quantize_per_channel_affine(weight, self.weight_scale, self.weight_zero, 0, QMIN, QMAX)
        return torch.fake_quantize_per_tensor_affine(weight, self.weight_scale, self.weight_zero, QMIN, QMAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qat and self.training:
            self._follow(x)
        x = torch.fake_quantize_per_tensor_affine(x, self.input_scale, self.input_zero, QMIN, QMAX)
        w = self.quantized_weight()
        layer = self.layer
        if isinstance(layer, nn.Conv2d):
            return F.conv2d(x, w, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
        return F.linear(x, w, layer.bias)


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    """What is quantized and how; also the record published with the evaluation."""

    mode: str
    granularity: str
    calibration: str | None
    regions: tuple[str, ...]
    input_ranges: Mapping[str, float] = field(default_factory=dict)
    calibration_n: int = 0
    calibration_sha256: str | None = None

    def definition(self) -> dict[str, Any]:
        return {
            "policy": QUANTIZATION_POLICY,
            "mode": self.mode,
            "granularity": self.granularity,
            "calibration": self.calibration,
            "regions": list(self.regions),
            "calibration_examples": self.calibration_n,
            "calibration_indices_sha256": self.calibration_sha256,
            "layers": len(self.input_ranges),
            "input_ranges": {k: round(float(v), 6) for k, v in sorted(self.input_ranges.items())},
        }


def int8_regions(stem: str, blocks: Sequence[str], head: str) -> tuple[str, ...]:
    """Names (``stem``, ``blocks.<i>``, ``head``) of the regions requested in INT8."""
    out = ["stem"] if stem == "int8" else []
    out += [f"blocks.{i}" for i, p in enumerate(blocks) if p == "int8"]
    return tuple(out + (["head"] if head == "int8" else []))


def plan_quantization(
    folded: nn.Module,
    regions: Sequence[str],
    images: torch.Tensor,
    mode: str,
    granularity: str,
    calibration: str,
    indices: Sequence[int],
) -> QuantizationPlan:
    """Calibrate the FP32 folded network on ``images`` (the calibration examples) for the INT8 regions."""
    if mode not in MODES:
        raise QuantizationError("unknown_mode", f"{mode!r} is not one of {MODES}")
    layers = quantizable_layers(folded, regions)
    if not layers:
        raise QuantizationError("no_layers", f"regions {list(regions)} have no convolution or linear layer")
    ranges = calibrate(folded, layers, images, calibration)
    return QuantizationPlan(
        mode, granularity, calibration, tuple(regions), ranges, len(indices), indices_sha256(indices)
    )


def insert_qdq(net: nn.Module, plan: QuantizationPlan) -> list[str]:
    """Replace, in place, the layers named in ``plan.input_ranges`` by their Q/DQ versions; returns their names."""
    done: list[str] = []
    for name, input_range in plan.input_ranges.items():
        parent_path, _, leaf = name.rpartition(".")
        parent = net.get_submodule(parent_path) if parent_path else net
        layer = getattr(parent, leaf)
        if isinstance(layer, QDQLayer):
            raise QuantizationError("already_quantized", name)
        setattr(parent, leaf, QDQLayer(layer, input_range, plan.granularity, qat=plan.mode == "qat"))
        done.append(name)
    return done
