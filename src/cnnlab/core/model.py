"""Construcao da rede convolucional usada pelo laboratorio."""

from __future__ import annotations

from torch import nn

from cnnlab.config import ModelConfig


def build_cnn(cfg: ModelConfig, num_classes: int) -> nn.Module:
    """Constroi uma CNN parametrizada por ``cfg``."""
    layers: list[nn.Module] = []
    in_channels = 3

    for block_index in range(cfg.n_blocks):
        out_channels = cfg.base_filters * 2**block_index
        layers.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(cfg.dropout * 0.5),
            ]
        )
        in_channels = out_channels

    layers.extend(
        [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(in_channels, num_classes),
        ]
    )

    model = nn.Sequential(*layers)
    _initialize_weights(model)
    return model


def count_params(model: nn.Module) -> int:
    """Retorna o numero total de parametros do modelo."""
    return sum(parameter.numel() for parameter in model.parameters())


def _initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
