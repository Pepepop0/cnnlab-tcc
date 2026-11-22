"""CIFAR-100 residente na VRAM.

Decisao D5 da arquitetura: o dataset inteiro e convertido para tensor uma unica
vez, movido para o device e fatiado na mao. Nao existe `DataLoader` neste
projeto — o overhead por batch domina o tempo quando o AG faz dezenas de
treinos curtos.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.datasets import CIFAR100

from cnnlab.config import DatasetConfig

CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)

# float32 e nao float16: com AMP ligado o autocast converte a entrada sozinho,
# e com AMP desligado um tensor float16 quebraria contra pesos float32. O
# dataset inteiro em float32 ocupa ~740 MB, dentro do orcamento da D5.
_STORAGE_DTYPE = torch.float32

# Seed do split train/val. Constante de proposito: todos os individuos do AG
# precisam ser avaliados sobre exatamente a mesma validacao, senao a fitness
# nao e comparavel entre eles.
_SPLIT_SEED = 20260901


@dataclass
class DataBundle:
    """Dataset inteiro ja normalizado e residente no device."""

    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    class_names: list[str]
    num_classes: int


_CACHE: dict[str, DataBundle] = {}


def load_data(cfg: DatasetConfig, device: torch.device) -> DataBundle:
    """Carrega CIFAR-100 como tensores no device, com cache em modulo.

    A segunda chamada com a mesma config e o mesmo device devolve o bundle ja
    carregado, sem tocar no disco nem baixar nada.
    """
    device = torch.device(device)
    key = _cache_key(cfg, device)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    bundle = _build_bundle(cfg, device)
    _CACHE[key] = bundle
    return bundle


def clear_data_cache() -> None:
    """Esvazia o cache de modulo. Existe para testes e para liberar VRAM."""
    _CACHE.clear()


def iter_batches(
    x: Tensor,
    y: Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    augment_flip: bool = False,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Percorre `x`/`y` em batches, sem sair do device onde eles ja estao.

    Os indices e a mascara de flip sao sorteados uma vez por epoca a partir de
    `generator`; dentro do laco de batches nao ha transferencia host->device.
    """
    n = int(x.shape[0])
    if n == 0:
        return
    if batch_size < 1:
        raise ValueError(f"batch_size deve ser >= 1, recebido {batch_size}")

    gen_device = generator.device
    channels_last = x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)

    if shuffle:
        order = torch.randperm(n, generator=generator, device=gen_device)
        order = order.to(x.device, non_blocking=True)
    else:
        order = None

    if augment_flip:
        flip = torch.rand(n, generator=generator, device=gen_device) < 0.5
        flip = flip.to(x.device, non_blocking=True)
    else:
        flip = None

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        # BatchNorm em modo treino quebra com batch de 1 amostra. Descartar a
        # sobra so no caminho de treino (shuffle=True); a validacao le tudo.
        if shuffle and stop - start == 1 and n > 1:
            return

        if order is None:
            xb = x[start:stop]
            yb = y[start:stop]
        else:
            idx = order[start:stop]
            xb = x[idx]
            yb = y[idx]

        if channels_last and not xb.is_contiguous(memory_format=torch.channels_last):
            xb = xb.contiguous(memory_format=torch.channels_last)

        if flip is not None:
            mask = flip[start:stop]
            xb = torch.where(mask.view(-1, 1, 1, 1), torch.flip(xb, dims=[3]), xb)

        yield xb, yb


def _cache_key(cfg: DatasetConfig, device: torch.device) -> str:
    return f"{cfg.model_dump_json()}|{device}"


def _build_bundle(cfg: DatasetConfig, device: torch.device) -> DataBundle:
    """Baixa (se preciso), normaliza, filtra classes e faz o split."""
    dataset = CIFAR100(root=str(cfg.root), train=True, download=True)

    x_u8 = torch.from_numpy(dataset.data)  # (N, 32, 32, 3) uint8
    y = torch.as_tensor(dataset.targets, dtype=torch.int64)
    class_names = list(dataset.classes)

    if cfg.class_subset is not None:
        x_u8, y, class_names = _take_first_classes(x_u8, y, class_names, cfg.class_subset)
    num_classes = len(class_names)

    train_idx, val_idx = _stratified_split(y, cfg.val_fraction, num_classes)

    target = device if cfg.cache_on_device else torch.device("cpu")
    x = _to_normalized(x_u8, target)
    y = y.to(target)

    train_idx = train_idx.to(x.device)
    val_idx = val_idx.to(x.device)

    return DataBundle(
        x_train=x[train_idx].contiguous(memory_format=torch.channels_last),
        y_train=y[train_idx],
        x_val=x[val_idx].contiguous(memory_format=torch.channels_last),
        y_val=y[val_idx],
        class_names=class_names,
        num_classes=num_classes,
    )


def _take_first_classes(
    x_u8: Tensor, y: Tensor, class_names: list[str], n_classes: int
) -> tuple[Tensor, Tensor, list[str]]:
    """Mantem so as `n_classes` primeiras classes e remapeia os rotulos."""
    kept = list(range(n_classes))
    remap = torch.full((len(class_names),), -1, dtype=torch.int64)
    for new_label, old_label in enumerate(kept):
        remap[old_label] = new_label

    keep = remap[y] >= 0
    return x_u8[keep], remap[y[keep]], [class_names[i] for i in kept]


def _stratified_split(y: Tensor, val_fraction: float, num_classes: int) -> tuple[Tensor, Tensor]:
    """Split estratificado do split de treino oficial. Deterministico."""
    gen = torch.Generator().manual_seed(_SPLIT_SEED)
    train_parts: list[Tensor] = []
    val_parts: list[Tensor] = []

    for label in range(num_classes):
        idx = (y == label).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        n_val = min(max(1, round(idx.numel() * val_fraction)), idx.numel() - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = torch.cat(train_parts)
    val_idx = torch.cat(val_parts)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=gen)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=gen)]
    return train_idx, val_idx


def _to_normalized(x_u8: Tensor, device: torch.device) -> Tensor:
    """uint8 NHWC na CPU -> float NCHW normalizado no device, em channels_last."""
    x = x_u8.to(device, non_blocking=True).permute(0, 3, 1, 2)
    x = x.contiguous(memory_format=torch.channels_last).to(_STORAGE_DTYPE).div_(255.0)
    mean = torch.tensor(CIFAR100_MEAN, dtype=_STORAGE_DTYPE, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=_STORAGE_DTYPE, device=device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)
