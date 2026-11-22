"""Well-known CIFAR networks through the final pipeline, outside the search space (exploratory, no test access).

ResNet-20/56, MobileNetV2 x0.5, ShuffleNetV2 x1 and SqueezeNet 1.1, in their usual CIFAR forms, FP32 and unpruned.
They are not genomes, so a subclass of ``FinalBuilder`` replaces only the step that makes the network; training, the
3 builds and the bench are the finalists' own. Only the validation split is used: the official test is never granted.
Run with PYTHONPATH=<frozen worktree>/src.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn

from cnnlab.tcc.model_pipeline import final_eval
from cnnlab.tcc.model_pipeline.builder import conv_bn, count_macs, count_parameters, fold_batchnorm
from cnnlab.tcc.model_pipeline.data_access import SplitIndices, TorchvisionCifar100, grant_search_access
from cnnlab.tcc.model_pipeline.engine import export_network, summarize_layers
from cnnlab.tcc.model_pipeline.evaluator import _Onnx
from cnnlab.tcc.model_pipeline.final_eval import FinalBuilder, FinalConfig, prepare_final_data
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec, poc_stratified_split
from cnnlab.tcc.model_pipeline.trainer import evaluate_accuracy, run_in_batches, train_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "piloto"))
import trava  # noqa: E402

CLASSES = 100
SEED = 20261022  # the seed of the two in-space references
LOG10_LR, LOG10_WD = -2.75, -4.0  # the learning rate and weight decay of the two in-space references


def relu() -> nn.Module:
    return nn.ReLU(inplace=False)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(*conv_bn(cin, cout, 3, stride), relu(), *conv_bn(cout, cout, 3))
        self.short = nn.Sequential(*conv_bn(cin, cout, 1, stride)) if stride != 1 or cin != cout else nn.Identity()
        self.out = relu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.main(x) + self.short(x))


class PooledNet(nn.Module):
    """Features, global average pooling and a linear classifier."""

    def __init__(self, features: nn.Module, width: int) -> None:
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(width, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).mean(dim=(2, 3)))


def resnet(depth: int) -> nn.Module:
    n = (depth - 2) // 6
    layers: list[nn.Module] = [*conv_bn(3, 16, 3), relu()]
    cin = 16
    for cout, stride in ((16, 1), (32, 2), (64, 2)):
        for i in range(n):
            layers.append(ResidualBlock(cin, cout, stride if i == 0 else 1))
            cin = cout
    return PooledNet(nn.Sequential(*layers), 64)


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expand: int) -> None:
        super().__init__()
        mid = cin * expand
        self.residual = stride == 1 and cin == cout
        self.main = nn.Sequential(
            *conv_bn(cin, mid, 1), relu(), *conv_bn(mid, mid, 3, stride, mid), relu(), *conv_bn(mid, cout, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x) if self.residual else self.main(x)


def mobilenetv2(width: float) -> nn.Module:
    cin = int(32 * width)
    layers: list[nn.Module] = [*conv_bn(3, cin, 3, 2), relu()]
    for t, c, n, s in (
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ):
        cout = int(c * width)
        for i in range(n):
            layers.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
            cin = cout
    layers += [*conv_bn(cin, 1280, 1), relu()]
    return PooledNet(nn.Sequential(*layers), 1280)


def shuffle(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    return x.view(n, 2, c // 2, h, w).transpose(1, 2).reshape(n, c, h, w)


class ShuffleUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.half = channels // 2
        self.right = nn.Sequential(
            *conv_bn(self.half, self.half, 1),
            relu(),
            *conv_bn(self.half, self.half, 3, 1, self.half),
            *conv_bn(self.half, self.half, 1),
            relu(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shuffle(torch.cat((x[:, : self.half], self.right(x[:, self.half :])), dim=1))


class ShuffleDown(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        mid = cout // 2
        self.left = nn.Sequential(*conv_bn(cin, cin, 3, 2, cin), *conv_bn(cin, mid, 1), relu())
        self.right = nn.Sequential(
            *conv_bn(cin, mid, 1), relu(), *conv_bn(mid, mid, 3, 2, mid), *conv_bn(mid, mid, 1), relu()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shuffle(torch.cat((self.left(x), self.right(x)), dim=1))


def shufflenetv2() -> nn.Module:
    layers: list[nn.Module] = [*conv_bn(3, 24, 1), relu()]
    cin = 24
    for cout, n in ((116, 3), (232, 7), (464, 3)):
        layers.append(ShuffleDown(cin, cout))
        layers += [ShuffleUnit(cout) for _ in range(n)]
        cin = cout
    layers += [*conv_bn(cin, 1024, 1), relu()]
    return PooledNet(nn.Sequential(*layers), 1024)


def squeezenet() -> nn.Module:
    model = torchvision.models.squeezenet1_1(weights=None, num_classes=CLASSES)
    model.features[0] = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # 32x32 images: the ImageNet stem would discard them
    return model


NETWORKS: dict[str, Callable[[], nn.Module]] = {
    "resnet20": lambda: resnet(20),
    "resnet56": lambda: resnet(56),
    "mobilenetv2-x05": lambda: mobilenetv2(0.5),
    "shufflenetv2-x1": shufflenetv2,
    "squeezenet11": squeezenet,
}


def check_fp32(_onnx: Path, inspector: Mapping[str, Any], *_regions: Any) -> dict[str, Any]:
    """The genome's per-region check has no regions here: the engine just has to hold no reduced precision."""
    dtypes = summarize_layers(dict(inspector))["output_dtypes"]
    bad = [d for d in dtypes if any(mark in d for mark in ("Half", "FP16", "BF16", "Int8", "INT8"))]
    return {"ok": not bad, "problems": [f"reduced precision in the engine: {d}" for d in bad], "output_dtypes": dtypes}


class ExternalBuilder(FinalBuilder):
    """``FinalBuilder`` whose network comes from a factory; only the step before the engine differs."""

    def __init__(self, *args: Any, factory: Callable[[], nn.Module], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.factory = factory

    def _make_onnx(
        self,
        expressed: Mapping[str, Any],
        train_seed: int,
        work: Path,
        durations: dict[str, float],
        quality: dict[str, Any],
        progress: dict[str, str],
        keep_networks: bool = False,
    ) -> _Onnx:
        progress["stage"] = "build"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(train_seed)
            model = self.factory()
        model = model.to(self.device)
        progress["stage"] = "train"
        t = time.monotonic()
        report = train_model(
            model,
            self.data.x_train,
            self.data.y_train,
            log10_lr=LOG10_LR,
            log10_wd=LOG10_WD,
            config=self.config.train,
            seed=train_seed,
        )
        durations["train"] = time.monotonic() - t
        quality["train"] = asdict(report)
        quality.setdefault("excluded_from_gpu_seconds", {})["paused"] = float(report.extra.get("paused_s", 0.0))
        quality["model"] = {"parameters": count_parameters(model), "macs": count_macs(model)}
        progress["stage"] = "reference"
        quality["torch_val_accuracy_diagnostic"] = evaluate_accuracy(model, self.data.x_val, self.data.y_val)
        folded = fold_batchnorm(model).float().eval()

        def logits(start: int, stop: int) -> torch.Tensor:
            with torch.no_grad():
                return folded(self.data.x_val[start:stop]).float().cpu()

        ref_path = work / "ref_logits.npy"
        np.save(ref_path, torch.cat(run_in_batches(logits, self.data.x_val.shape[0])).numpy())
        progress["stage"] = "export"
        t = time.monotonic()
        exported = export_network(folded, work)
        durations["export"] = time.monotonic() - t
        kept = (folded.cpu(), folded.cpu()) if keep_networks else None
        del model
        self._release()
        return _Onnx(
            exported=exported,
            ref_path=ref_path,
            block_precisions=[],
            regions={"stem": "fp32", "head": "fp32"},
            genome=None,
            networks=kept,
        )  # type: ignore[arg-type]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cifar-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    final_eval.check_engine_precision = check_fp32
    trava.take("referencias-externas", "redes conhecidas (validacao, sem teste)", wait_s=60)
    try:
        source = TorchvisionCifar100(args.cifar_root)
        _, labels = source.read("train")
        spec = DemoSplitSpec(train_per_class=None, subset_seed=0)
        train_idx, val_idx = poc_stratified_split(labels, spec.val_fraction, spec.split_seed)
        grant = grant_search_access(
            "referencias-externas",
            "pilot",
            [SplitIndices("train", "train", tuple(train_idx)), SplitIndices("val", "train", tuple(val_idx))],
        )
        data = prepare_final_data(grant, source, torch.device("cuda"))
        config = FinalConfig(epochs=args.epochs, builds=args.builds)
        for name, factory in NETWORKS.items():
            if args.only and name not in args.only:
                continue
            if (args.out / f"{name}.json").exists():
                print(f"{name}: ja feito, pulando", flush=True)
                continue
            work = args.out / f"work-{name}"
            builder = ExternalBuilder(
                data, work / "val.npz", config, device="cuda", work_root=work / "work", factory=factory
            )
            t0 = time.monotonic()
            built = builder.run({}, SEED, args.out / "art", name)
            row = {
                "name": name,
                "status": built.status,
                "e_val": built.e_val,
                "acc_val": None if built.e_val is None else 1 - built.e_val,
                "l_ms": built.l_ms,
                "v_mb": built.v_mb,
                "s_mb": built.s_mb,
                "model": (built.quality or {}).get("model"),
                "torch_val_accuracy": (built.quality or {}).get("torch_val_accuracy_diagnostic"),
                "builds_l_ms": [b.l_ms for b in built.builds],
                "wall_s": round(time.monotonic() - t0),
                "epochs": args.epochs,
                "failure": [built.failure_stage, built.failure_code, built.failure_detail],
            }
            (args.out / f"{name}.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            print(
                f"{name}: {built.status} acc_val={row['acc_val']} L={built.l_ms} V={built.v_mb} S={built.s_mb} "
                f"params={(row['model'] or {}).get('parameters')} ({row['wall_s']} s) {row['failure']}",
                flush=True,
            )
        return 0
    finally:
        trava.release()


if __name__ == "__main__":
    sys.exit(main())
