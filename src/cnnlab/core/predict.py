"""Carregamento de modelos e classificacao de imagens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from cnnlab.config import ModelConfig
from cnnlab.core.data import CIFAR100_MEAN, CIFAR100_STD, DataBundle
from cnnlab.core.model import build_cnn
from cnnlab.storage import layout
from cnnlab.storage.reader import artifact_path


@dataclass
class Prediction:
    predicted: str
    confidence: float
    top5: list[tuple[str, float]]


def load_model(run_id: str, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Reconstroi um modelo a partir do checkpoint seguro de uma run."""
    path = artifact_path(run_id)
    if path is None:
        raise FileNotFoundError(f"artifact not found for run {run_id!r}")

    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    model_config = ModelConfig(**payload["model_config"])
    class_names = list(payload["class_names"])
    model = build_cnn(model_config, num_classes=len(class_names))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    # Metadado simples, nao faz parte do state_dict nem do checkpoint.
    model._cnnlab_run_id = run_id  # type: ignore[attr-defined]
    return model, class_names


def predict_tensor(model: nn.Module, x: Tensor, class_names: list[str]) -> Prediction:
    """Classifica uma imagem CHW ou um batch unitario NCHW ja normalizado."""
    prediction = _predict(model, x, class_names)
    _record_prediction(model, prediction, image=None, source="tensor")
    return prediction


def predict_file(
    model: nn.Module,
    path: Path,
    class_names: list[str],
    device: torch.device,
) -> Prediction:
    """Abre, prepara e classifica uma imagem externa."""
    path = Path(path)
    with Image.open(path) as image:
        image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)

    x = pixels.reshape(32, 32, 3).permute(2, 0, 1).to(torch.float32).div_(255.0)

    # Fonte da verdade: CIFAR100_MEAN/STD em core/data.py. Estes valores devem
    # permanecer identicos aos usados no treino.
    mean = torch.tensor(CIFAR100_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=x.dtype).view(3, 1, 1)
    x = x.sub_(mean).div_(std).to(device)

    prediction = _predict(model, x, class_names)
    _record_prediction(model, prediction, image=str(path), source="upload")
    return prediction


def sample_from_dataset(data: DataBundle, index: int | None = None) -> tuple[Tensor, str]:
    """Seleciona uma imagem do conjunto de validacao e seu nome de classe."""
    sample_count = len(data.x_val)
    if sample_count == 0:
        raise ValueError("validation dataset is empty")
    if index is None:
        index = int(torch.randint(sample_count, ()).item())

    label = int(data.y_val[index].item())
    return data.x_val[index], data.class_names[label]


def _predict(model: nn.Module, x: Tensor, class_names: list[str]) -> Prediction:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("x must have shape (C, H, W) or (1, C, H, W)")

    model.eval()
    with torch.no_grad():
        logits = model(x)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model must return one logits vector")
        if logits.shape[1] != len(class_names):
            raise ValueError("class_names length must match the model output")
        probabilities = torch.softmax(logits[0], dim=0)
        values, indices = probabilities.topk(min(5, len(class_names)))

    top5 = [(class_names[int(index)], float(value)) for value, index in zip(values.cpu(), indices.cpu(), strict=True)]
    return Prediction(
        predicted=top5[0][0],
        confidence=top5[0][1],
        top5=top5,
    )


def _record_prediction(
    model: nn.Module,
    prediction: Prediction,
    *,
    image: str | None,
    source: str,
) -> None:
    run_id = getattr(model, "_cnnlab_run_id", None)
    if not isinstance(run_id, str):
        return

    payload: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "image": image,
        "source": source,
        "predicted": prediction.predicted,
        "confidence": prediction.confidence,
        "top5": prediction.top5,
    }
    layout.append_jsonl(layout.PREDICTIONS_FILE, payload)
