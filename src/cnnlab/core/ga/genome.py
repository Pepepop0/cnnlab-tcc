"""Representacao do genoma e operadores geneticos.

O genoma e um `dict`, nao uma lista: o `.jsonl` precisa ser legivel por humano
e legibilidade vale mais que elegancia de encoding nesta escala.

`lr` e `weight_decay` sao amostrados em log10 mas guardados ja convertidos para
o valor linear. Mutacao e clamp convertem de volta, perturbam e voltam.

Toda aleatoriedade entra por um `random.Random` recebido como parametro. Nada
de `random` global: reprodutibilidade e requisito (decisao D7).
"""

from __future__ import annotations

import math
import random
from typing import Any

from cnnlab.config import ModelConfig, OptimConfig, SearchSpace, TrainConfig

Genome = dict[str, Any]

_CONTINUOUS: tuple[str, ...] = ("lr", "weight_decay", "dropout")
_CATEGORICAL: tuple[str, ...] = ("batch_size", "optimizer", "base_filters", "n_blocks")
GENE_KEYS: tuple[str, ...] = _CONTINUOUS + _CATEGORICAL

# Genes amostrados em log10. O dicionario guarda o valor linear.
_LOG_KEYS = frozenset({"lr", "weight_decay"})

# Atributo do SearchSpace que guarda a faixa de cada gene continuo.
_RANGE_ATTR = {
    "lr": "lr_log10",
    "weight_decay": "weight_decay_log10",
    "dropout": "dropout",
}


def sample(space: SearchSpace, rng: random.Random) -> Genome:
    """Sorteia um genoma uniformemente dentro do espaco de busca."""
    genome: Genome = {}
    for key in _CONTINUOUS:
        lo, hi = _bounds(space, key)
        genome[key] = _decode(key, rng.uniform(lo, hi))
    for key in _CATEGORICAL:
        genome[key] = rng.choice(_options(space, key))
    return genome


def to_train_config(g: Genome, epochs: int, seed: int, amp: bool, channels_last: bool) -> TrainConfig:
    """Traduz genoma para `TrainConfig`.

    `seed` e a mesma para todos os individuos de propósito (regra 4 da spec):
    com 3 epocas, ruido de inicializacao dominaria a diferenca de fitness.
    """
    return TrainConfig(
        model=ModelConfig(
            base_filters=g["base_filters"],
            n_blocks=g["n_blocks"],
            dropout=g["dropout"],
        ),
        optim=OptimConfig(
            optimizer=g["optimizer"],
            lr=g["lr"],
            weight_decay=g["weight_decay"],
            batch_size=g["batch_size"],
        ),
        epochs=epochs,
        amp=amp,
        channels_last=channels_last,
        seed=seed,
    )


def crossover(a: Genome, b: Genome, rng: random.Random) -> tuple[Genome, Genome]:
    """Crossover uniforme: cada gene troca de lado com probabilidade 0.5."""
    child_a = dict(a)
    child_b = dict(b)
    for key in GENE_KEYS:
        if rng.random() < 0.5:
            child_a[key], child_b[key] = child_b[key], child_a[key]
    return child_a, child_b


def mutate(g: Genome, space: SearchSpace, sigma: float, rng: random.Random) -> Genome:
    """Perturba genes independentemente. Sempre muda pelo menos um.

    A probabilidade por gene e `1 / len(GENE_KEYS)` — a convencao de "um gene
    por mutacao, em media". O engine ja decidiu mutar este individuo antes de
    chamar aqui, entao uma chamada que nao mudasse nada seria desperdicio de
    orcamento: se o sorteio nao pegar nenhum gene, um e escolhido a mao.
    """
    mutant = dict(g)
    per_gene = 1.0 / len(GENE_KEYS)
    touched = [key for key in GENE_KEYS if rng.random() < per_gene]
    if not touched:
        touched = [rng.choice(GENE_KEYS)]

    for key in touched:
        if key in _CONTINUOUS:
            lo, hi = _bounds(space, key)
            value = _encode(space, key, mutant[key])
            value += rng.gauss(0.0, sigma * (hi - lo))
            mutant[key] = _decode(key, min(max(value, lo), hi))
        else:
            mutant[key] = _other_option(mutant[key], _options(space, key), rng)

    return mutant


def clamp(g: Genome, space: SearchSpace) -> Genome:
    """Devolve um genoma garantidamente dentro do `SearchSpace`.

    Chame sempre depois de crossover e mutate: `SearchSpace` e mais estreito
    que os limites do pydantic, e um genoma fora da faixa passaria na validacao
    do `TrainConfig` mas sujaria a busca.
    """
    fixed = dict(g)
    for key in _CONTINUOUS:
        lo, hi = _bounds(space, key)
        value = _encode(space, key, fixed[key])
        fixed[key] = _decode(key, min(max(value, lo), hi))
    for key in _CATEGORICAL:
        fixed[key] = _nearest_option(fixed[key], _options(space, key))
    return fixed


def _bounds(space: SearchSpace, key: str) -> tuple[float, float]:
    lo, hi = getattr(space, _RANGE_ATTR[key])
    return float(lo), float(hi)


def _options(space: SearchSpace, key: str) -> list[Any]:
    return list(getattr(space, key))


def _encode(space: SearchSpace, key: str, value: float) -> float:
    """Valor linear -> espaco de amostragem. Zero cai no piso da faixa."""
    if key in _LOG_KEYS:
        lo, _ = _bounds(space, key)
        return math.log10(value) if value > 0.0 else lo
    return float(value)


def _decode(key: str, value: float) -> float:
    """Espaco de amostragem -> valor linear guardado no genoma."""
    return 10.0**value if key in _LOG_KEYS else float(value)


def _other_option(current: Any, options: list[Any], rng: random.Random) -> Any:
    """Escolhe outro valor da lista, uniformemente. Lista de 1 fica parada."""
    alternatives = [option for option in options if option != current]
    if not alternatives:
        return current
    return rng.choice(alternatives)


def _nearest_option(value: Any, options: list[Any]) -> Any:
    """Snap de um categorico invalido: mais proximo se numerico, senao o 1o."""
    if value in options:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = [o for o in options if isinstance(o, (int, float))]
        if numeric:
            return min(numeric, key=lambda option: abs(option - value))
    return options[0]
