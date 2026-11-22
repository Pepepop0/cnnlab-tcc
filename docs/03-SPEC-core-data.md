# Spec — core/data.py

## Responsabilidade

Carregar CIFAR-100 e entregar tensores prontos para o trainer, residentes na
GPU. Nada de `DataLoader`.

## Interface

```python
@dataclass
class DataBundle:
    x_train: Tensor  # (N, 3, 32, 32) float16 ou float32, ja normalizado, em device
    y_train: Tensor  # (N,) int64, em device
    x_val: Tensor
    y_val: Tensor
    class_names: list[str]
    num_classes: int


def load_data(cfg: DatasetConfig, device: torch.device) -> DataBundle: ...


def iter_batches(
    x: Tensor,
    y: Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    augment_flip: bool = False,
) -> Iterator[tuple[Tensor, Tensor]]: ...
```

## Regras

1. Baixa via `torchvision.datasets.CIFAR100(root=cfg.root, download=True)`.
   Faz isso uma vez; depois le do disco.
2. Converte o dataset inteiro para tensor de uma vez, normaliza com media e
   desvio do CIFAR-100, e move para `device`. Nao usa `DataLoader`.
   - mean = (0.5071, 0.4865, 0.4409)
   - std  = (0.2673, 0.2564, 0.2762)
3. Se `cfg.class_subset` for um inteiro N, filtra para as N primeiras classes e
   remapeia os rotulos para 0..N-1. `num_classes` reflete o valor efetivo.
4. Split train/val a partir do split de treino oficial, estratificado, com
   `cfg.val_fraction`. O split de teste do CIFAR-100 nao e usado na POC.
5. `iter_batches` gera indices com `torch.randperm(..., generator=generator)`
   e fatia. Nada de transferencia host para device dentro do loop.
6. Augmentation e apenas flip horizontal, feito em GPU com
   `torch.flip(batch, dims=[3])` numa mascara aleatoria de metade do batch.
   Aplicar somente quando `augment_flip=True`, e nunca na validacao.
7. `load_data` deve ser cacheavel: se chamado duas vezes com a mesma config,
   pode reaproveitar. Use um cache em modulo com chave no dump da config.

## Riscos

- Se `cache_on_device=False`, mantenha em CPU e mova por batch. Caminho de
  fallback para OOM, nao o padrao.
- Verifique `x.is_contiguous(memory_format=torch.channels_last)` quando
  `channels_last` estiver ligado no trainer.

## Aceite

- [ ] `load_data` retorna tensores em `cuda` e nao dispara download na 2a chamada
- [ ] `class_subset=10` retorna `num_classes == 10` e rotulos em 0..9
- [ ] `iter_batches` com mesma seed produz a mesma ordem duas vezes
- [ ] Um epoch completo nao faz nenhuma transferencia host para device
