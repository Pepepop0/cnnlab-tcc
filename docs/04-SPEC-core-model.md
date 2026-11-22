# Spec — core/model.py

## Responsabilidade

Construir a CNN a partir de `ModelConfig`. Uma familia unica, parametrizada.
Sem selecao de arquitetura (ver decisao D6).

## Interface

```python
def build_cnn(cfg: ModelConfig, num_classes: int) -> nn.Module: ...
def count_params(model: nn.Module) -> int: ...
```

## Arquitetura

Bloco convolucional repetido `n_blocks` vezes, com largura dobrando a cada
bloco a partir de `base_filters`:

```
bloco i:
  Conv2d(in, out, kernel_size=3, padding=1, bias=False)
  BatchNorm2d(out)
  ReLU(inplace=True)
  Conv2d(out, out, kernel_size=3, padding=1, bias=False)
  BatchNorm2d(out)
  ReLU(inplace=True)
  MaxPool2d(2)
  Dropout2d(cfg.dropout * 0.5)

cabeca:
  AdaptiveAvgPool2d(1)
  Flatten()
  Dropout(cfg.dropout)
  Linear(final_width, num_classes)
```

Larguras: `base_filters * 2**i` para o bloco i. Com `base_filters=32` e
`n_blocks=3`: 32, 64, 128.

Entrada 32x32. Apos 3 MaxPool: 4x4. `AdaptiveAvgPool2d(1)` torna a cabeca
independente de `n_blocks`, o que evita recalcular a dimensao do `Linear` — e
o motivo dele existir aqui em vez de um flatten direto.

## Regras

1. `build_cnn` e puro: mesma config, mesma topologia. Nao move para device,
   nao aplica `channels_last`. Quem faz isso e o trainer.
2. Inicializacao: `kaiming_normal_` nas conv, `constant_` 1/0 nas batchnorm.
3. Nao registrar hooks, nao guardar estado global.

## Aceite

- [ ] `build_cnn(ModelConfig(), 100)` faz forward em `(2,3,32,32)` e sai `(2,100)`
- [ ] `n_blocks=2` e `n_blocks=4` funcionam sem mudar a cabeca
- [ ] `count_params` bate com `sum(p.numel() for p in model.parameters())`
