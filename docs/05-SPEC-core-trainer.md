# Spec — core/trainer.py

## Responsabilidade

Treinar um modelo por N epocas, emitir `EpochEnd` a cada epoca, respeitar
cancelamento, devolver metricas. Nao sabe o que e um AG. Nao sabe o que e a TUI.

## Interface

```python
@dataclass
class TrainResult:
    best_val_acc: float
    final_val_acc: float
    final_train_loss: float
    epochs_run: int
    elapsed_s: float
    cancelled: bool
    state_dict: dict | None  # so quando keep_weights=True


@dataclass
class UnitContext:
    """Onde este treino se encaixa na run. O AG preenche gen/individual."""

    run_id: str
    unit_index: int
    unit_total: int
    gen: int | None = None
    individual: int | None = None


def train(
    train_cfg: TrainConfig,
    data: DataBundle,
    ctx: UnitContext,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
    keep_weights: bool = False,
) -> TrainResult: ...
```

## Regras

1. Semear com `train_cfg.seed` no inicio: `torch.manual_seed`, `random.seed`,
   `numpy.random.seed`. Criar um `torch.Generator` dedicado para os batches.
2. Ligar `torch.backends.cudnn.benchmark = True` uma vez, no import do modulo.
3. AMP quando `train_cfg.amp`: `torch.amp.autocast("cuda", dtype=torch.float16)`
   mais `torch.amp.GradScaler()`. Use a API nova, nao `torch.cuda.amp`.
4. `channels_last` quando ligado: aplicar em modelo e batch de entrada.
5. Otimizador conforme `optim.optimizer`:
   - `adam`: `torch.optim.Adam(lr=..., weight_decay=...)`
   - `sgd_momentum`: `torch.optim.SGD(lr=..., momentum=0.9, weight_decay=..., nesterov=True)`
6. Loss: `nn.CrossEntropyLoss()`.
7. Ao fim de cada epoca: avaliar em validacao com `torch.no_grad()` e
   `model.eval()`, emitir `EpochEnd`, voltar a `model.train()`.
8. Checar `cancel.is_set()` no fim de cada epoca. Se setado, parar, marcar
   `cancelled=True` e retornar o que tiver. Nao levantar excecao.
9. `keep_weights=True` retorna `state_dict` movido para CPU. O AG so pede isso
   para o melhor individuo, para nao encher a RAM.
10. Tratar OOM: capturar `torch.cuda.OutOfMemoryError`, chamar
    `torch.cuda.empty_cache()`, tentar uma vez com metade do batch. Se falhar
    de novo, propagar.

## Antipadroes

- Nao chamar `on_event` dentro do loop de batches. Uma emissao por epoca.
- Nao acumular historico de tensores. Metrica e float.
- Nao usar `print`. Quem reporta e o `on_event`.

## Aceite

- [ ] Treino de 2 epocas emite exatamente 2 `EpochEnd`
- [ ] Duas execucoes com a mesma seed dao a mesma `final_val_acc`
- [ ] `cancel.set()` durante a epoca 1 retorna com `cancelled=True` e `epochs_run==1`
- [ ] AMP ligado reduz o tempo por epoca em relacao a desligado
