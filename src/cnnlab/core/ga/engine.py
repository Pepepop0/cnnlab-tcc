"""Loop evolutivo.

Quatro invariantes seguram a corretude deste modulo:

1. Elites carregam genoma **e** fitness, e nao sao retreinadas. E isso que faz
   `GAConfig.budget()` bater com a contagem real de chamadas a `train`.
2. Toda aleatoriedade sai de `random.Random(job.ga.seed)`.
3. O `state_dict` final vem de **um retreino** do melhor genoma no fim, nao de
   pesos guardados durante a busca. Mantem a VRAM previsivel.
4. Todos os individuos recebem a mesma seed de inicializacao: a variacao de
   fitness tem que vir do genoma, nao do acaso.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cnnlab.config import GAConfig, GAJob, SearchSpace
from cnnlab.core.data import DataBundle
from cnnlab.core.ga.genome import Genome, clamp, crossover, mutate, sample, to_train_config
from cnnlab.core.trainer import TrainResult, UnitContext, train
from cnnlab.events import (
    Event,
    GenerationDone,
    IndividualDone,
    RunFinished,
    RunStarted,
)

# Nome do arquivo de pesos. Duplicado do layout de disco de proposito: `core`
# nao pode importar `storage` (regra de dependencia da arquitetura).
_ARTIFACT_NAME = "best.pt"


@dataclass
class GAResult:
    best_genome: Genome
    best_fitness: float
    best_state_dict: dict | None
    generations_run: int
    total_trainings: int
    cancelled: bool
    # Acuracia do retreino final, que e a acuracia real de `best_state_dict`.
    # `best_fitness` mede as poucas epocas da busca e subestima o peso salvo:
    # quem for rotular o `best.pt` tem que usar este campo. `None` quando nao
    # houve retreino (run cancelada).
    final_val_acc: float | None = None


@dataclass
class _Individual:
    """Genoma mais o fitness ja medido. `None` significa "ainda nao avaliado"."""

    genome: Genome
    fitness: float | None = None


def run_ga(
    job: GAJob,
    data: DataBundle,
    run_id: str,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
) -> GAResult:
    """Roda o algoritmo genetico e devolve o melhor genoma com seus pesos.

    Em cancelamento, retorna o melhor encontrado ate ali sem `state_dict` e sem
    emitir `RunFinished` — quem decide o evento terminal da run e o worker.
    """
    cfg = job.ga
    space = cfg.search_space
    rng = random.Random(cfg.seed)
    trainings, _ = cfg.budget()
    started = time.perf_counter()

    on_event(
        RunStarted(
            run_id=run_id,
            kind="ga",
            unit_total=trainings,
            epochs_per_unit=cfg.epochs_per_individual,
        )
    )

    population = [_Individual(sample(space, rng)) for _ in range(cfg.population_size)]
    unit = 0
    generations_run = 0
    best = _Individual({}, None)

    for gen in range(cfg.generations):
        gen_started = time.perf_counter()

        for index, individual in enumerate(population):
            # Elite carregada da geracao anterior: fitness ja medido, pula.
            if individual.fitness is not None:
                continue

            individual.fitness, unit = _evaluate(
                individual.genome,
                job,
                data,
                run_id,
                gen,
                index,
                unit,
                trainings,
                on_event,
                cancel,
                device,
            )

            if cancel.is_set():
                return GAResult(
                    best_genome=dict(best.genome),
                    best_fitness=best.fitness or 0.0,
                    best_state_dict=None,
                    generations_run=generations_run,
                    total_trainings=unit,
                    cancelled=True,
                )

        scores = [i.fitness for i in population if i.fitness is not None]
        champion = max(population, key=_fitness_of)
        if best.fitness is None or champion.fitness > best.fitness:
            best = _Individual(dict(champion.genome), champion.fitness)

        on_event(
            GenerationDone(
                run_id=run_id,
                gen=gen,
                best_fitness=max(scores),
                mean_fitness=statistics.fmean(scores),
                std_fitness=statistics.pstdev(scores),
                best_genome=dict(champion.genome),
                elapsed_s=time.perf_counter() - gen_started,
            )
        )
        generations_run = gen + 1

        if gen == cfg.generations - 1:
            break
        population = _next_population(population, cfg, space, rng)

    retrain = _retrain_champion(best, job, data, run_id, trainings, on_event, cancel, device)
    state_dict = retrain.state_dict if retrain is not None else None
    # `final_val_acc` e nao `best_val_acc`: o trainer devolve os pesos da ultima
    # epoca, nao os da melhor, entao e a acuracia final que descreve o que foi
    # salvo.
    final_val_acc = retrain.final_val_acc if retrain is not None else None

    on_event(
        RunFinished(
            run_id=run_id,
            kind="ga",
            best_fitness=best.fitness or 0.0,
            best_genome=dict(best.genome),
            total_trainings=unit,
            total_epochs=unit * cfg.epochs_per_individual,
            elapsed_s=time.perf_counter() - started,
            artifact=_ARTIFACT_NAME if state_dict is not None else None,
        )
    )

    return GAResult(
        best_genome=dict(best.genome),
        best_fitness=best.fitness or 0.0,
        best_state_dict=state_dict,
        generations_run=generations_run,
        total_trainings=unit,
        cancelled=cancel.is_set(),
        final_val_acc=final_val_acc,
    )


def _evaluate(
    genome: Genome,
    job: GAJob,
    data: DataBundle,
    run_id: str,
    gen: int,
    index: int,
    unit: int,
    trainings: int,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
) -> tuple[float, int]:
    """Treina um individuo. Devolve (fitness, proximo unit)."""
    cfg = job.ga
    train_cfg = to_train_config(
        genome,
        epochs=cfg.epochs_per_individual,
        seed=cfg.seed,
        amp=job.amp,
        channels_last=job.channels_last,
    )
    ctx = UnitContext(
        run_id=run_id,
        unit_index=unit,
        unit_total=trainings,
        gen=gen,
        individual=index,
    )

    individual_started = time.perf_counter()
    result = train(train_cfg, data, ctx, on_event, cancel, device, keep_weights=False)

    on_event(
        IndividualDone(
            run_id=run_id,
            gen=gen,
            individual=index,
            genome=dict(genome),
            fitness=result.best_val_acc,
            epochs=result.epochs_run,
            elapsed_s=time.perf_counter() - individual_started,
        )
    )
    return result.best_val_acc, unit + 1


def _next_population(
    population: list[_Individual],
    cfg: GAConfig,
    space: SearchSpace,
    rng: random.Random,
) -> list[_Individual]:
    """Elites intactas (com fitness) mais filhos ainda por avaliar."""
    ranked = sorted(population, key=_fitness_of, reverse=True)
    elites = [_Individual(dict(i.genome), i.fitness) for i in ranked[: cfg.elitism]]

    slots = cfg.population_size - cfg.elitism
    children: list[Genome] = []
    while len(children) < slots:
        parent_a = _tournament(population, cfg.tournament_size, rng)
        parent_b = _tournament(population, cfg.tournament_size, rng)

        if rng.random() < cfg.crossover_prob:
            child_a, child_b = crossover(parent_a.genome, parent_b.genome, rng)
        else:
            child_a, child_b = dict(parent_a.genome), dict(parent_b.genome)

        for child in (child_a, child_b):
            if rng.random() < cfg.mutation_prob:
                child = mutate(child, space, cfg.mutation_sigma, rng)
            children.append(clamp(child, space))

    return elites + [_Individual(genome) for genome in children[:slots]]


def _tournament(population: list[_Individual], size: int, rng: random.Random) -> _Individual:
    """Torneio com reposicao.

    Escrito a mao em vez de `deap.tools.selTournament`: a funcao do DEAP sorteia
    pelo modulo `random` global e nao aceita um `Random` proprio, o que quebraria
    a reprodutibilidade por seed. Ver docs/DECISIONS.md.
    """
    best = rng.choice(population)
    for _ in range(size - 1):
        challenger = rng.choice(population)
        if _fitness_of(challenger) > _fitness_of(best):
            best = challenger
    return best


def _retrain_champion(
    best: _Individual,
    job: GAJob,
    data: DataBundle,
    run_id: str,
    trainings: int,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
) -> TrainResult | None:
    """Retreina o melhor genoma so para extrair o `state_dict`.

    Roda por `final_epochs`, nao pelas poucas epocas da busca: e um treino so,
    e e ele que alimenta a tela de predicao. Com 3 epocas o `best.pt` sairia
    inutil.

    Ocupa o slot da ultima unidade no progresso (`unit_index = trainings - 1`):
    `RunStarted` ja declarou `unit_total`, entao um indice novo estouraria a
    barra. Nao conta em `total_trainings`, que existe para bater com `budget()`.
    """
    if not best.genome or cancel.is_set():
        return None

    train_cfg = to_train_config(
        best.genome,
        epochs=job.ga.final_epochs,
        seed=job.ga.seed,
        amp=job.amp,
        channels_last=job.channels_last,
    )
    ctx = UnitContext(
        run_id=run_id,
        unit_index=max(trainings - 1, 0),
        unit_total=trainings,
        gen=job.ga.generations - 1,
        individual=0,
    )
    return train(train_cfg, data, ctx, on_event, cancel, device, keep_weights=True)


def _fitness_of(individual: _Individual) -> float:
    """Nao avaliado perde qualquer torneio."""
    return individual.fitness if individual.fitness is not None else float("-inf")
