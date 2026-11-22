"""Testes de carregamento e inferencia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from cnnlab.config import ModelConfig
from cnnlab.core import predict as predict_module
from cnnlab.core.data import CIFAR100_MEAN, CIFAR100_STD, DataBundle
from cnnlab.core.model import build_cnn
from cnnlab.core.predict import (
    load_model,
    predict_file,
    predict_tensor,
    sample_from_dataset,
)


class RecordingModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))
        self.seen_x: Tensor | None = None
        self.grad_enabled: bool | None = None
        self.training_seen: bool | None = None

    def forward(self, x: Tensor) -> Tensor:
        self.seen_x = x.detach().clone()
        self.grad_enabled = torch.is_grad_enabled()
        self.training_seen = self.training
        return self.logits.unsqueeze(0)


def test_load_model_rebuilds_checkpoint_without_serialized_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(base_filters=8, n_blocks=2, dropout=0.1)
    class_names = ["apple", "orange", "pear"]
    original = build_cnn(config, len(class_names))
    artifact = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "model_config": config.model_dump(),
            "class_names": class_names,
        },
        artifact,
    )
    monkeypatch.setattr(predict_module, "artifact_path", lambda _run_id: artifact)

    loaded, loaded_names = load_model("run-1", torch.device("cpu"))

    assert loaded_names == class_names
    assert not loaded.training
    assert loaded._cnnlab_run_id == "run-1"
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_load_model_raises_when_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predict_module, "artifact_path", lambda _run_id: None)

    with pytest.raises(FileNotFoundError):
        load_model("missing", torch.device("cpu"))


def test_predict_tensor_uses_eval_no_grad_softmax_and_top5() -> None:
    model = RecordingModel([1.0, 5.0, 2.0, 4.0, 3.0])
    model.train()
    class_names = ["a", "b", "c", "d", "e"]

    prediction = predict_tensor(model, torch.zeros(3, 32, 32), class_names)

    assert prediction.predicted == "b"
    assert prediction.confidence == pytest.approx(torch.softmax(model.logits, 0)[1].item())
    assert len(prediction.top5) == 5
    assert sum(confidence for _, confidence in prediction.top5) == pytest.approx(1.0)
    assert model.training_seen is False
    assert model.grad_enabled is False


def test_predict_file_uses_training_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "solid.png"
    Image.new("RGB", (48, 20), color=(255, 128, 0)).save(path)
    model = RecordingModel([2.0, 1.0])

    prediction = predict_file(
        model,
        path,
        ["first", "second"],
        torch.device("cpu"),
    )

    expected_channels = torch.tensor(
        [
            (1.0 - CIFAR100_MEAN[0]) / CIFAR100_STD[0],
            ((128 / 255) - CIFAR100_MEAN[1]) / CIFAR100_STD[1],
            (0.0 - CIFAR100_MEAN[2]) / CIFAR100_STD[2],
        ]
    )
    assert prediction.predicted == "first"
    assert model.seen_x is not None
    assert model.seen_x.shape == (1, 3, 32, 32)
    torch.testing.assert_close(
        model.seen_x.mean(dim=(0, 2, 3)),
        expected_channels,
        rtol=1e-5,
        atol=1e-7,
    )
    assert model.training_seen is False
    assert model.grad_enabled is False


def test_sample_from_dataset_returns_selected_validation_item() -> None:
    data = DataBundle(
        x_train=torch.empty(0, 3, 32, 32),
        y_train=torch.empty(0, dtype=torch.int64),
        x_val=torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)]),
        y_val=torch.tensor([1, 0]),
        class_names=["zero", "one"],
        num_classes=2,
    )

    sample, class_name = sample_from_dataset(data, index=1)

    assert torch.equal(sample, data.x_val[1])
    assert class_name == "zero"


def test_prediction_is_appended_to_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(predict_module.layout, "PREDICTIONS_FILE", history)
    model = RecordingModel([3.0, 1.0])
    model._cnnlab_run_id = "run-history"

    predict_tensor(model, torch.zeros(3, 32, 32), ["cat", "dog"])

    record = json.loads(history.read_text(encoding="utf-8"))
    assert record["run_id"] == "run-history"
    assert record["source"] == "tensor"
    assert record["predicted"] == "cat"
    assert len(record["top5"]) == 2
