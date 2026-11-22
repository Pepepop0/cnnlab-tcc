"""Testes de core/data.py.

Os testes rapidos trocam `CIFAR100` por um fake em memoria: cobrem contrato e
determinismo sem baixar 170 MB. Os que exigem o dataset real ou GPU estao
marcados como `slow`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cnnlab.config import DatasetConfig
from cnnlab.core import data as data_mod
from cnnlab.core.data import DataBundle, clear_data_cache, iter_batches, load_data

N_PER_CLASS = 20
N_CLASSES = 100


class FakeCIFAR100:
    """Substituto de `torchvision.datasets.CIFAR100` com a mesma superficie usada."""

    instantiations = 0

    def __init__(self, root: str, train: bool = True, download: bool = False) -> None:
        type(self).instantiations += 1
        rng = np.random.default_rng(0)
        self.data = rng.integers(0, 256, size=(N_CLASSES * N_PER_CLASS, 32, 32, 3), dtype=np.uint8)
        self.targets = [c for c in range(N_CLASSES) for _ in range(N_PER_CLASS)]
        self.classes = [f"class_{i:03d}" for i in range(N_CLASSES)]


@pytest.fixture(autouse=True)
def _fake_dataset(monkeypatch: pytest.MonkeyPatch):
    FakeCIFAR100.instantiations = 0
    clear_data_cache()
    monkeypatch.setattr(data_mod, "CIFAR100", FakeCIFAR100)
    yield
    clear_data_cache()


def _bundle(**kwargs) -> DataBundle:
    return load_data(DatasetConfig(**kwargs), torch.device("cpu"))


# --- load_data ---------------------------------------------------------------


def test_shapes_dtypes_and_device() -> None:
    b = _bundle(val_fraction=0.1)

    assert b.x_train.shape[1:] == (3, 32, 32)
    assert b.x_val.shape[1:] == (3, 32, 32)
    assert b.x_train.dtype is torch.float32
    assert b.y_train.dtype is torch.int64
    assert b.x_train.shape[0] == b.y_train.shape[0]
    assert b.x_val.shape[0] == b.y_val.shape[0]
    assert b.x_train.shape[0] + b.x_val.shape[0] == N_CLASSES * N_PER_CLASS
    assert b.x_train.device.type == "cpu"


def test_normalization_uses_cifar100_stats() -> None:
    b = _bundle()
    # Dados uniformes tem media 0.5 por canal; depois da normalizacao a media
    # esperada e (0.5 - mean) / std.
    expected = [(0.5 - m) / s for m, s in zip(data_mod.CIFAR100_MEAN, data_mod.CIFAR100_STD)]
    got = b.x_train.mean(dim=(0, 2, 3)).tolist()
    assert got == pytest.approx(expected, abs=0.05)


def test_tensors_are_channels_last() -> None:
    b = _bundle()
    assert b.x_train.is_contiguous(memory_format=torch.channels_last)
    assert b.x_val.is_contiguous(memory_format=torch.channels_last)


def test_class_subset_filters_and_remaps() -> None:
    b = _bundle(class_subset=10)

    assert b.num_classes == 10
    assert len(b.class_names) == 10
    assert int(b.y_train.min()) == 0
    assert int(b.y_train.max()) == 9
    assert int(b.y_val.max()) <= 9
    assert b.x_train.shape[0] + b.x_val.shape[0] == 10 * N_PER_CLASS


def test_split_is_stratified() -> None:
    b = _bundle(class_subset=10, val_fraction=0.2)
    counts = torch.bincount(b.y_val, minlength=10)
    assert counts.tolist() == [round(N_PER_CLASS * 0.2)] * 10


def test_split_is_deterministic_across_calls() -> None:
    a = _bundle(class_subset=5)
    clear_data_cache()
    b = _bundle(class_subset=5)
    assert torch.equal(a.y_val, b.y_val)
    assert torch.equal(a.x_val, b.x_val)


def test_second_call_is_cached_and_does_not_reload() -> None:
    first = _bundle(class_subset=5)
    second = _bundle(class_subset=5)

    assert second is first
    assert FakeCIFAR100.instantiations == 1


def test_different_config_gets_its_own_cache_entry() -> None:
    _bundle(class_subset=5)
    other = _bundle(class_subset=6)

    assert other.num_classes == 6
    assert FakeCIFAR100.instantiations == 2


def test_cache_on_device_false_keeps_tensors_on_cpu() -> None:
    b = load_data(DatasetConfig(cache_on_device=False, class_subset=5), torch.device("cpu"))
    assert b.x_train.device.type == "cpu"


# --- iter_batches ------------------------------------------------------------


def _toy(n: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n * 3 * 32 * 32, dtype=torch.float32).reshape(n, 3, 32, 32)
    x = x.contiguous(memory_format=torch.channels_last)
    return x, torch.arange(n, dtype=torch.int64)


def test_no_shuffle_yields_dataset_in_order() -> None:
    x, y = _toy(50)
    gen = torch.Generator().manual_seed(0)
    batches = list(iter_batches(x, y, batch_size=16, shuffle=False, generator=gen))

    assert [b[0].shape[0] for b in batches] == [16, 16, 16, 2]
    assert torch.equal(torch.cat([b[1] for b in batches]), y)


def test_same_seed_gives_same_order_twice() -> None:
    x, y = _toy(50)

    def order(seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.cat([b[1] for b in iter_batches(x, y, 8, shuffle=True, generator=gen)])

    assert torch.equal(order(123), order(123))
    assert not torch.equal(order(123), order(456))


def test_shuffle_is_a_permutation_without_repeats() -> None:
    x, y = _toy(48)
    gen = torch.Generator().manual_seed(7)
    got = torch.cat([b[1] for b in iter_batches(x, y, 8, shuffle=True, generator=gen)])

    assert torch.equal(got.sort().values, y)


def test_x_and_y_stay_aligned_under_shuffle() -> None:
    x, y = _toy(32)
    gen = torch.Generator().manual_seed(3)
    for xb, yb in iter_batches(x, y, 8, shuffle=True, generator=gen):
        assert torch.equal(xb, x[yb])


def test_trailing_batch_of_one_is_dropped_only_when_shuffling() -> None:
    x, y = _toy(17)

    gen = torch.Generator().manual_seed(1)
    shuffled = list(iter_batches(x, y, 8, shuffle=True, generator=gen))
    assert sum(b[0].shape[0] for b in shuffled) == 16

    gen = torch.Generator().manual_seed(1)
    ordered = list(iter_batches(x, y, 8, shuffle=False, generator=gen))
    assert sum(b[0].shape[0] for b in ordered) == 17


def test_augment_flip_off_returns_untouched_samples() -> None:
    x, y = _toy(16)
    gen = torch.Generator().manual_seed(0)
    batches = list(iter_batches(x, y, 16, shuffle=False, generator=gen))

    assert torch.equal(batches[0][0], x)


def test_augment_flip_flips_about_half_horizontally() -> None:
    x, y = _toy(200)
    gen = torch.Generator().manual_seed(11)
    xb, _ = next(iter_batches(x, y, 200, shuffle=False, generator=gen, augment_flip=True))

    flipped = (xb == torch.flip(x, dims=[3])).all(dim=3).all(dim=2).all(dim=1)
    same = (xb == x).all(dim=3).all(dim=2).all(dim=1)
    assert bool((flipped | same).all())
    assert 60 <= int(flipped.sum()) <= 140  # ~metade, com folga estatistica


def test_augment_flip_is_reproducible_and_does_not_mutate_source() -> None:
    x, y = _toy(64)
    original = x.clone()

    def run() -> torch.Tensor:
        gen = torch.Generator().manual_seed(99)
        return torch.cat([b[0] for b in iter_batches(x, y, 16, shuffle=False, generator=gen, augment_flip=True)])

    assert torch.equal(run(), run())
    assert torch.equal(x, original)


def test_batches_keep_channels_last() -> None:
    x, y = _toy(40)
    gen = torch.Generator().manual_seed(5)
    for xb, _ in iter_batches(x, y, 16, shuffle=True, generator=gen, augment_flip=True):
        assert xb.is_contiguous(memory_format=torch.channels_last)


def test_empty_input_yields_nothing() -> None:
    x, y = _toy(50)
    gen = torch.Generator().manual_seed(0)
    assert list(iter_batches(x[:0], y[:0], 8, shuffle=True, generator=gen)) == []


def test_invalid_batch_size_raises() -> None:
    x, y = _toy(8)
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        list(iter_batches(x, y, 0, shuffle=False, generator=gen))


# --- GPU / dataset real ------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="exige GPU CUDA")


@pytest.mark.slow
@requires_cuda
def test_bundle_lives_on_cuda_and_batches_never_leave_it() -> None:
    b = load_data(DatasetConfig(class_subset=5), torch.device("cuda"))
    assert b.x_train.device.type == "cuda"
    assert b.y_train.device.type == "cuda"

    gen = torch.Generator(device="cuda").manual_seed(0)
    for xb, yb in iter_batches(b.x_train, b.y_train, 128, shuffle=True, generator=gen, augment_flip=True):
        assert xb.device.type == "cuda"
        assert yb.device.type == "cuda"


@pytest.mark.slow
@requires_cuda
def test_cpu_generator_also_works_on_cuda_tensors() -> None:
    b = load_data(DatasetConfig(class_subset=5), torch.device("cuda"))
    gen = torch.Generator().manual_seed(0)
    xb, yb = next(iter_batches(b.x_train, b.y_train, 64, shuffle=True, generator=gen, augment_flip=True))
    assert xb.device.type == "cuda" and yb.device.type == "cuda"


@pytest.mark.slow
def test_real_cifar100_downloads_once_and_matches_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baixa CIFAR-100 de verdade. Rode com `pytest -m slow`."""
    monkeypatch.undo()  # desfaz o fake instalado pelo fixture
    clear_data_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DatasetConfig(class_subset=10, val_fraction=0.1)
    b = load_data(cfg, device)

    assert b.num_classes == 10
    assert b.x_train.shape[0] + b.x_val.shape[0] == 5000
    assert int(b.y_train.max()) == 9
    assert load_data(cfg, device) is b
