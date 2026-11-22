"""Testes da construcao da CNN."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cnnlab.config import ModelConfig
from cnnlab.core.model import build_cnn, count_params


@pytest.mark.parametrize("n_blocks", [2, 3, 4])
def test_build_cnn_forward_for_supported_block_counts(n_blocks: int) -> None:
    num_classes = 17
    model = build_cnn(ModelConfig(n_blocks=n_blocks), num_classes)

    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, num_classes)


def test_build_cnn_uses_adaptive_pooling_before_classifier() -> None:
    model = build_cnn(ModelConfig(base_filters=16, n_blocks=3), num_classes=5)
    modules = list(model.children())

    assert isinstance(modules[-4], nn.AdaptiveAvgPool2d)
    assert isinstance(modules[-3], nn.Flatten)
    assert isinstance(modules[-2], nn.Dropout)
    assert isinstance(modules[-1], nn.Linear)
    assert modules[-1].in_features == 64
    assert modules[-1].out_features == 5


def test_count_params_counts_every_model_parameter() -> None:
    model = build_cnn(ModelConfig(), num_classes=100)

    assert count_params(model) == sum(parameter.numel() for parameter in model.parameters())


def test_build_cnn_initializes_convolutions_and_batch_norms() -> None:
    model = build_cnn(ModelConfig(), num_classes=10)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert not torch.count_nonzero(module.weight).eq(0)
        elif isinstance(module, nn.BatchNorm2d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
