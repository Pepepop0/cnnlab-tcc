"""Per-region precision: materialize FP32/FP16/BF16/INT8 regions in the graph and confirm what the engine runs.

The regions are the stem, each block and the head. The network input and the
logits are always FP32, so a stem or head that is not FP32 gets a cast at its
outer boundary.

TCC-02 rule 4: the precision of a block has to be *materialized in the graph,
with casts at the boundaries*, and the engine inspector has to confirm the
type that is actually executed; a requested precision that does not show up
makes the candidate invalid.

* :func:`typed_network` folds BN in PyTorch, converts the weights of every
  FP16 region to half precision and inserts a cast only where the precision
  changes between consecutive regions (and at the input and the logits).
* :func:`check_engine_precision` reads the exported ONNX and the TensorRT
  inspector. Every convolution and the classifier layers are identified by
  the block their weights belong to (``blocks.<i>.…``). The inspector names,
  in ``Metadata``, the ONNX nodes behind each engine layer, fused ones
  included; that layer must consume and produce the requested type. When
  TensorRT fused a boundary cast into the layer, the side of the cast has
  the neighbour's type and only the other side is compared. When casts were
  fused on both sides, neither side shows the layer's own type; the layer is
  then accepted only if the weights stored in the engine have the requested
  type, and the row says so (``evidence = "engine weights"``). On 29/09/2026
  TensorRT 11.3 fused at most one cast per layer in 24 one-layer FP16 blocks
  between FP32 neighbours, so this last rule has not been seen at work.

``fp32``, ``fp16`` and ``bf16`` are materialized by casts. ``int8`` is
materialized by explicit Q/DQ on the input and the weights of each layer of
the region (``quantization.py``); its activations between layers stay FP32.
For an INT8 region the check wants Q/DQ in the graph and Int8 in the engine
layer's input and weights. Pruning is not implemented here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cnnlab.tcc.model_pipeline.builder import TCCNet, fold_batchnorm
from cnnlab.tcc.model_pipeline.quantization import MODES, QuantizationPlan, insert_qdq, int8_regions

# What the typed network can build and the check can read.
# An INT8 region computes its activations in FP32 between layers; Q/DQ nodes carry the quantization.
TORCH_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.float32,
}
ONNX_DTYPES: dict[str, str] = {"fp32": "FLOAT", "fp16": "FLOAT16", "bf16": "BFLOAT16", "int8": "INT8"}
ENGINE_DTYPES: dict[str, str] = {"fp32": "Float", "fp16": "Half", "bf16": "BFloat16", "int8": "Int8"}
ADMITTED_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8")
CAST_PRECISION: dict[str, str] = {"int8": "fp32"}  # what an INT8 region looks like to the casts around it
# TensorRT 11.3 (RTX 5070), measured by `precision_probe --matrix bf16` (evidence/wave3-g2-*): HardSwish does not
# build in BF16, and depthwise convolutions in a BF16 region run as FP16. Such candidates are invalid, not
# approximated (a silent FP16 substitution would label the wrong precision on the measurement).
BF16_UNSUPPORTED_ACTIVATIONS: tuple[str, ...] = ("hardswish",)
BF16_DEPTHWISE_OPS: tuple[str, ...] = ("dw_separable", "mbconv_e3", "mbconv_e6")
SUPPORTED_BLOCK_PRECISIONS: tuple[str, ...] = ADMITTED_PRECISIONS
SUPPORTED_REGION_PRECISIONS: tuple[str, ...] = ADMITTED_PRECISIONS
WEIGHT_OPS: tuple[str, ...] = ("Conv", "Gemm", "MatMul")
PRECISION_POLICY = "tcc-precision/per-region-fp32-fp16-bf16-int8-v4"
_REGION = re.compile(r"^(stem|head|blocks\.(\d+))\.")
_ONNX_LAYER = re.compile(r"\[ONNX Layer: ([^\]]+)\]")


class TypedNet(nn.Module):
    """Folded network whose regions run in their own precision; casts only where the precision changes."""

    def __init__(
        self,
        folded: TCCNet,
        block_precisions: Sequence[str],
        stem_precision: str = "fp32",
        head_precision: str = "fp32",
        quantization: QuantizationPlan | None = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        if len(block_precisions) != len(folded.blocks):
            raise ValueError("one precision per block is required")
        unknown = sorted((set(block_precisions) | {stem_precision, head_precision}) - set(TORCH_DTYPES))
        if unknown:
            raise ValueError(f"precisions {unknown} are not materialized (buildable: {tuple(TORCH_DTYPES)})")
        wanted_int8 = int8_regions(stem_precision, block_precisions, head_precision)
        if wanted_int8 and (quantization is None or set(quantization.regions) != set(wanted_int8)):
            raise ValueError(f"INT8 regions {list(wanted_int8)} need a quantization plan calibrated for them")
        if quantization is not None and quantization.input_ranges and not quantized:
            insert_qdq(folded, quantization)  # ``quantized``: the network already carries its (QAT) Q/DQ layers
        self.quantization = quantization
        self.stem = folded.stem
        self.blocks = folded.blocks
        self.head = folded.head
        self.stem_precision = stem_precision
        self.head_precision = head_precision
        self.block_precisions = tuple(block_precisions)
        self.stem.to(TORCH_DTYPES[stem_precision])
        for block, precision in zip(self.blocks, self.block_precisions):
            block.to(TORCH_DTYPES[precision])
        self.head.to(TORCH_DTYPES[head_precision])

    @staticmethod
    def _as(x: torch.Tensor, precision: str) -> torch.Tensor:
        dtype = TORCH_DTYPES[precision]
        return x if x.dtype == dtype else x.to(dtype)  # INT8 regions take FP32, so no cast reaches them

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self._as(x, self.stem_precision))
        for block, precision in zip(self.blocks, self.block_precisions):
            x = block(self._as(x, precision))
        x = self.head(self._as(x, self.head_precision))
        return self._as(x, "fp32")

    @property
    def casts(self) -> int:
        """Number of precision boundaries, i.e. casts in the exported graph."""
        chain = ("fp32", self.stem_precision, *self.block_precisions, self.head_precision, "fp32")
        chain = tuple(CAST_PRECISION.get(p, p) for p in chain)
        return sum(a != b for a, b in zip(chain, chain[1:]))


def typed_network(
    model: TCCNet,
    block_precisions: Sequence[str],
    stem_precision: str = "fp32",
    head_precision: str = "fp32",
    quantization: QuantizationPlan | None = None,
    prefolded: bool = False,
) -> TypedNet:
    """Eval-mode copy of ``model`` with BN folded and each region in its precision; ``model`` is untouched.

    INT8 regions need the ``quantization`` plan (calibrated on the folded FP32 network); its layers get Q/DQ.
    With ``prefolded`` the network is already folded and carries its Q/DQ layers (QAT fine-tuned): it is used as
    it is, not copied.
    """
    if prefolded:
        return TypedNet(model, block_precisions, stem_precision, head_precision, quantization, quantized=True).eval()  # type: ignore[arg-type]
    return TypedNet(fold_batchnorm(model), block_precisions, stem_precision, head_precision, quantization).eval()  # type: ignore[arg-type]


def region_precisions(genome: Any) -> dict[str, Any]:
    """``{"stem", "blocks", "head"}`` precisions of an expressed genome, as the typed network takes them."""
    return {
        "stem": genome.stem_precision,
        "blocks": [b.precision for b in genome.blocks],
        "head": genome.head_precision,
    }


def representation_problems(genome: Any) -> list[str]:
    """Genes the pipeline cannot materialize yet; such a candidate is invalid, not approximated.

    FP32, FP16, BF16 and INT8 (post-training quantization or QAT) are
    materialized in every region, except the combinations the engine is known
    not to run (see ``bf16_engine_limits``). Structured pruning is applied by
    the evaluator (``pruning.py``), not here. Sparsity is not materialized
    (yet), so it is listed here instead of approximated.
    """
    problems = [
        f"{name}={getattr(genome, name)}"
        for name in ("stem_precision", "head_precision")
        if getattr(genome, name) not in SUPPORTED_REGION_PRECISIONS
    ]
    for i, block in enumerate(genome.blocks):
        if block.precision not in SUPPORTED_REGION_PRECISIONS:
            problems.append(f"blocks[{i}].precision={block.precision}")
        if block.sparsity not in (None, "dense"):
            problems.append(f"blocks[{i}].sparsity={block.sparsity}")
    if (
        int8_regions(genome.stem_precision, [b.precision for b in genome.blocks], genome.head_precision)
        and genome.int8_mode not in MODES
    ):
        problems.append(f"int8_mode={genome.int8_mode}")
    return problems + bf16_engine_limits(genome)


def bf16_engine_limits(genome: Any) -> list[str]:
    """BF16 requests that TensorRT 11.3 cannot honour for this network (checked before any build)."""
    problems: list[str] = []
    if genome.activation in BF16_UNSUPPORTED_ACTIVATIONS:
        regions = {"stem": genome.stem_precision, **{f"blocks[{i}]": b.precision for i, b in enumerate(genome.blocks)}}
        if genome.head == "gap_fc256_linear":  # only a head with a hidden layer has an activation to reject
            regions["head"] = genome.head_precision
        problems += [
            f"{name}.precision=bf16 with activation {genome.activation} (the engine rejects BF16 there)"
            for name, precision in regions.items()
            if precision == "bf16"
        ]
    problems += [
        f"blocks[{i}].precision=bf16 with op {b.op} (the engine runs its depthwise convolution as fp16)"
        for i, b in enumerate(genome.blocks)
        if b.precision == "bf16" and b.op in BF16_DEPTHWISE_OPS
    ]
    return problems


def region_of(initializer_name: str) -> str | None:
    """``stem``, ``head`` or ``blocks.<i>`` for a weight named after the module that owns it."""
    match = _REGION.match(initializer_name)
    return match.group(1) if match else None


def requested_precisions(
    block_precisions: Sequence[str], stem_precision: str = "fp32", head_precision: str = "fp32"
) -> dict[str, str]:
    out = {"stem": stem_precision, "head": head_precision}
    out.update({f"blocks.{i}": precision for i, precision in enumerate(block_precisions)})
    return out


def weight_layers(onnx_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Layers with weights in the exported graph, and the names of its Cast nodes."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializers = {t.name: onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer}
    producers = {out: node for node in model.graph.node for out in node.output}

    def behind_qdq(name: str) -> str | None:
        """The initializer that reaches ``name`` through DequantizeLinear (<- QuantizeLinear) or None."""
        dq = producers.get(name)
        if dq is None or dq.op_type != "DequantizeLinear":
            return None
        source = dq.input[0]
        q = producers.get(source)
        if q is not None and q.op_type == "QuantizeLinear":
            source = q.input[0]
        return source if source in initializers else None

    layers = []
    for node in model.graph.node:
        if node.op_type not in WEIGHT_OPS:
            continue
        weights = {name: name for name in node.input if name in initializers}  # plain: the weight is the input
        quantized = {name: behind_qdq(name) for name in node.input if behind_qdq(name)}
        weights |= quantized
        if not weights:
            continue
        # With Q/DQ the layer's weights are INT8 whatever the stored FLOAT initializer says; the bias stays FLOAT.
        dtypes = {"INT8"} if quantized else {initializers[source] for source in weights.values()}
        regions = {region_of(source) for source in weights.values()}
        data = producers.get(node.input[0])
        layers.append(
            {
                "node": node.name,
                "op": node.op_type,
                "region": next(iter(regions)) if len(regions) == 1 else None,
                "onnx_weight_dtypes": sorted(dtypes),
                "input_qdq": data is not None and data.op_type == "DequantizeLinear" and node.input[0] not in quantized,
            }
        )
    return layers, {node.name for node in model.graph.node if node.op_type == "Cast"}


def quantize_nodes(onnx_path: Path) -> set[str]:
    """Names of the QuantizeLinear nodes of the exported graph (the ones that quantize an activation or a weight)."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return {node.name for node in model.graph.node if node.op_type == "QuantizeLinear"}


def engine_layers_by_node(inspector: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """ONNX node name -> the engine layer built from it, read from the inspector's ``Metadata``."""
    index: dict[str, dict[str, Any]] = {}
    for layer in inspector.get("Layers", []):
        if not isinstance(layer, dict):
            continue
        nodes = _ONNX_LAYER.findall(str(layer.get("Metadata", "")))
        for node in nodes:
            index.setdefault(node, {**layer, "onnx_nodes": nodes})
    return index


COMPUTE_LAYER_TYPES = frozenset({"correlation", "gemm", "convolution", "matrix_multiply", "fully_connected"})


def engine_compute_layers(inspector: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """ONNX node name -> the engine layer that does its arithmetic (convolution, GEMM), not a helper kernel.

    TensorRT lists the same ONNX node on helper kernels too (the activation Q, the padding of a 3-channel
    stem), and those come first; the layer whose input and weights say what the arithmetic runs in is the other.
    """
    index: dict[str, dict[str, Any]] = {}
    for layer in inspector.get("Layers", []):
        if isinstance(layer, dict) and str(layer.get("LayerType")) in COMPUTE_LAYER_TYPES:
            nodes = _ONNX_LAYER.findall(str(layer.get("Metadata", "")))
            for node in nodes:
                index.setdefault(node, {**layer, "onnx_nodes": nodes})
    return index


def main_weight_dtype(layer: Mapping[str, Any]) -> str | None:
    """Type of the largest constant of an engine layer (its weights; scales and biases are much smaller)."""
    constants = layer.get("Constants") if isinstance(layer.get("Constants"), list) else []
    sized = [
        (int(np.prod(c.get("Dimensions") or [0])), str(c.get("Datatype")))
        for c in constants
        if isinstance(c, dict) and len(c.get("Dimensions") or []) >= 2
    ]
    return max(sized)[1] if sized else None


def check_engine_precision(
    onnx_path: Path,
    inspector: Mapping[str, Any],
    block_precisions: Sequence[str],
    stem_precision: str = "fp32",
    head_precision: str = "fp32",
) -> dict[str, Any]:
    """Compare, layer by layer, the requested precision with the ONNX weights and the engine's types.

    ``ok`` is False when a layer with weights cannot be attributed to a
    region, carries weights of another type, is missing from the engine, or
    runs with an input or output type other than the requested one.
    """
    requested = requested_precisions(block_precisions, stem_precision, head_precision)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    layers, casts = weight_layers(onnx_path)
    quantizers = quantize_nodes(onnx_path)
    by_node = engine_layers_by_node(inspector)
    compute_by_node = engine_compute_layers(inspector)
    if not by_node:
        problems.append("the engine inspector names no ONNX layer (build the engine with detailed profiling)")
    for layer in layers:
        row: dict[str, Any] = {
            **layer,
            "requested": None,
            "engine_layer": None,
            "engine_input": None,
            "engine_output": None,
            "engine_weights": None,
            "tactic": None,
            "fused_cast": False,
            "fused_casts": 0,
            "evidence": None,
        }
        rows.append(row)
        region = layer["region"]
        if region is None or region not in requested:
            problems.append(f"{layer['node']}: weights cannot be attributed to a region of the genome")
            continue
        precision = row["requested"] = requested[region]
        if layer["onnx_weight_dtypes"] != [ONNX_DTYPES[precision]]:
            problems.append(
                f"{layer['node']} ({region}): {precision} requested, ONNX weights are {layer['onnx_weight_dtypes']}"
            )
        found = by_node.get(layer["node"])
        if found is None:
            problems.append(f"{layer['node']} ({region}): no engine layer was built from this node")
            continue
        if precision == "int8":
            found = compute_by_node.get(layer["node"], found)
        inputs = [str(t.get("Datatype")) for t in found.get("Inputs", [])]
        outputs = [str(t.get("Datatype")) for t in found.get("Outputs", [])]
        fused = sum(node in casts for node in found["onnx_nodes"])
        fused_cast = fused > 0
        constants = found.get("Constants") if isinstance(found.get("Constants"), list) else []
        weights = sorted(
            {str(c.get("Datatype")) for c in constants if isinstance(c, dict) and len(c.get("Dimensions") or []) >= 2}
        )
        if precision == "int8":  # the Q/DQ scales are Float constants of the same layer: judge the weights alone
            weights = [main_weight_dtype(found)] if main_weight_dtype(found) else []
        row.update(
            engine_layer=str(found.get("Name")),
            engine_input=inputs[0] if inputs else None,
            engine_output=outputs[0] if outputs else None,
            tactic=str(found.get("TacticName", ""))[:80],
            fused_cast=fused_cast,
            fused_casts=fused,
            engine_weights=weights or None,
        )
        wanted = ENGINE_DTYPES[precision]
        sides = [row["engine_input"] == wanted, row["engine_output"] == wanted]
        if precision == "int8":
            # Explicit Q/DQ: the data input of the kernel is Int8 and so are its stored weights. The output is
            # Int8 only when another Q follows, so it is not required.
            if not layer.get("input_qdq"):
                problems.append(f"{layer['node']} ({region}): int8 requested, the graph has no Q/DQ on its input")
            elif sides[0] and weights == [wanted]:
                row["evidence"] = "int8 input and weights"
            else:
                problems.append(
                    f"{layer['node']} ({region}): int8 requested, engine runs {row['engine_input']} -> "
                    f"{row['engine_output']} with weights {weights or 'unknown'}"
                )
        elif all(sides):
            row["evidence"] = "input and output"
        elif fused_cast and any(sides):
            row["evidence"] = "one side, cast fused on the other"
        elif (
            row["engine_input"] == wanted
            and row["engine_output"] == "Int8"
            and any(node in quantizers for node in found["onnx_nodes"])
        ):
            # The activation Q of the INT8 region that follows was fused into this layer's epilogue: the arithmetic
            # runs in the requested type and only its result is requantized (observed: Half -> Int8 before INT8 blocks).
            row["evidence"] = "input; output requantized by a fused Q"
        elif fused >= 2 and weights == [wanted]:
            row["evidence"] = "engine weights"  # casts fused on both sides hide the layer's own type
        elif weights == [wanted] and row["engine_output"] == wanted:
            # The engine fused the input cast into the kernel (no ONNX node left to show it): weights and
            # result are in the requested type, so the multiply-accumulate is too.
            row["evidence"] = "engine weights and output"
        else:
            problems.append(
                f"{layer['node']} ({region}): {precision} requested, engine runs "
                f"{row['engine_input']} -> {row['engine_output']}"
                + (f" with {fused} cast(s) fused, engine weights {weights or 'unknown'}" if fused_cast else "")
            )
    by_region: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["region"] is None:
            continue
        entry = by_region.setdefault(
            row["region"], {"requested": row["requested"], "layers": 0, "fused_casts": 0, "executed": set()}
        )
        entry["layers"] += 1
        entry["fused_casts"] += int(row["fused_cast"])
        wanted = ENGINE_DTYPES.get(str(row["requested"]))
        seen = [t for t in (row["engine_input"], row["engine_output"]) if t]
        # With a fused boundary cast, the cast's side shows the neighbour's type; report the layer's own.
        if row["evidence"] in ("int8 input and weights", "input; output requantized by a fused Q"):
            entry["executed"].add(str(wanted))
        elif row["evidence"] in ("engine weights", "engine weights and output"):
            entry["executed"].add(str(wanted))
            if row["evidence"] == "engine weights":
                entry["by_weights_only"] = entry.get("by_weights_only", 0) + 1
        else:
            entry["executed"].update([t for t in seen if t == wanted] if row["fused_cast"] and wanted in seen else seen)
    missing = sorted(set(requested) - set(by_region))
    if missing:
        problems.append(f"regions without any layer with weights in the graph: {missing}")
    return {
        "policy": PRECISION_POLICY,
        "ok": not problems,
        "problems": problems,
        "requested": requested,
        "regions": {
            name: {**entry, "executed": sorted(entry["executed"])} for name, entry in sorted(by_region.items())
        },
        "layers": rows,
    }
