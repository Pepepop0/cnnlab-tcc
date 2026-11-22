"""Testes de core/ga/.

`train` e mockado por uma funcao que devolve fitness sintetico derivado do
genoma: o loop evolutivo inteiro roda em segundos, sem GPU e sem dataset.
"""

from __future__ import annotations

import math
import random
import threading

import pytest
import torch

from cnnlab.config import GAConfig, GAJob, SearchSpace
from cnnlab.core.ga import engine as engine_mod
from cnnlab.core.ga.engine import GAResult, run_ga
from cnnlab.core.ga.genome import (
    GENE_KEYS,
    clamp,
    crossover,
    mutate,
    sample,
    to_train_config,
)
from cnnlab.core.trainer import TrainResult
from cnnlab.events import GenerationDone, IndividualDone, RunFinished, RunStarted

SPACE = SearchSpace()


# --- genome.py ---------------------------------------------------------------


def test_sample_stays_inside_the_search_space() -> None:
    rng = random.Random(0)
    for _ in range(200):
        g = sample(SPACE, rng)
        assert set(g) == set(GENE_KEYS)
        assert 10 ** SPACE.lr_log10[0] <= g["lr"] <= 10 ** SPACE.lr_log10[1]
        assert 10 ** SPACE.weight_decay_log10[0] <= g["weight_decay"] <= 10 ** SPACE.weight_decay_log10[1]
        assert SPACE.dropout[0] <= g["dropout"] <= SPACE.dropout[1]
        assert g["batch_size"] in SPACE.batch_size
        assert g["optimizer"] in SPACE.optimizer
        assert g["base_filters"] in SPACE.base_filters
        assert g["n_blocks"] in SPACE.n_blocks


def test_sample_is_reproducible_by_seed() -> None:
    a = [sample(SPACE, random.Random(7)) for _ in range(3)]
    b = [sample(SPACE, random.Random(7)) for _ in range(3)]
    assert a == b
    assert a != [sample(SPACE, random.Random(8)) for _ in range(3)]


def test_sample_ignores_the_global_random() -> None:
    random.seed(1)
    a = sample(SPACE, random.Random(99))
    random.seed(2)
    b = sample(SPACE, random.Random(99))
    assert a == b


def test_to_train_config_maps_every_gene() -> None:
    g = sample(SPACE, random.Random(3))
    cfg = to_train_config(g, epochs=3, seed=42, amp=True, channels_last=False)

    assert cfg.model.base_filters == g["base_filters"]
    assert cfg.model.n_blocks == g["n_blocks"]
    assert cfg.model.dropout == pytest.approx(g["dropout"])
    assert cfg.optim.optimizer == g["optimizer"]
    assert cfg.optim.lr == pytest.approx(g["lr"])
    assert cfg.optim.weight_decay == pytest.approx(g["weight_decay"])
    assert cfg.optim.batch_size == g["batch_size"]
    assert (cfg.epochs, cfg.seed, cfg.amp, cfg.channels_last) == (3, 42, True, False)


def test_crossover_only_swaps_genes_between_the_parents() -> None:
    rng = random.Random(5)
    a = sample(SPACE, rng)
    b = sample(SPACE, rng)
    child_a, child_b = crossover(a, b, rng)

    for key in GENE_KEYS:
        assert {child_a[key], child_b[key]} == {a[key], b[key]}
    assert a == sample(SPACE, random.Random(5))  # pais intactos


def test_crossover_actually_swaps_something_over_many_draws() -> None:
    rng = random.Random(11)
    a = {**sample(SPACE, rng), "dropout": 0.0}
    b = {**sample(SPACE, rng), "dropout": 0.5}
    swaps = sum(crossover(a, b, rng)[0]["dropout"] == 0.5 for _ in range(200))
    assert 60 < swaps < 140  # ~50%


def test_mutate_changes_at_least_one_gene_and_respects_bounds() -> None:
    rng = random.Random(13)
    for _ in range(200):
        g = sample(SPACE, rng)
        m = mutate(g, SPACE, sigma=0.15, rng=rng)
        assert m != g
        assert m == clamp(m, SPACE)


def test_mutate_does_not_touch_the_original() -> None:
    rng = random.Random(2)
    g = sample(SPACE, rng)
    snapshot = dict(g)
    mutate(g, SPACE, 0.3, rng)
    assert g == snapshot


def test_mutate_of_a_categorical_picks_a_different_value() -> None:
    rng = random.Random(4)
    g = sample(SPACE, rng)
    g["optimizer"] = "adam"
    changed = [mutate(g, SPACE, 0.15, rng)["optimizer"] for _ in range(300)]
    assert "sgd_momentum" in changed


def test_clamp_pulls_out_of_range_values_back() -> None:
    broken = {
        "lr": 5.0,  # acima de 10**-2
        "weight_decay": 0.0,  # log10 indefinido
        "dropout": 9.0,
        "batch_size": 1000,
        "optimizer": "rmsprop",
        "base_filters": 7,
        "n_blocks": 99,
    }
    fixed = clamp(broken, SPACE)

    assert fixed["lr"] == pytest.approx(10 ** SPACE.lr_log10[1])
    assert fixed["weight_decay"] == pytest.approx(10 ** SPACE.weight_decay_log10[0])
    assert fixed["dropout"] == pytest.approx(SPACE.dropout[1])
    assert fixed["batch_size"] == max(SPACE.batch_size)
    assert fixed["optimizer"] in SPACE.optimizer
    assert fixed["base_filters"] == min(SPACE.base_filters)
    assert fixed["n_blocks"] == max(SPACE.n_blocks)


def test_clamp_is_idempotent_and_keeps_valid_genomes() -> None:
    rng = random.Random(6)
    g = sample(SPACE, rng)
    assert clamp(g, SPACE) == g
    assert clamp(clamp(g, SPACE), SPACE) == g


def test_clamped_genome_always_builds_a_valid_train_config() -> None:
    rng = random.Random(21)
    for _ in range(100):
        g = mutate(sample(SPACE, rng), SPACE, sigma=0.9, rng=rng)
        to_train_config(clamp(g, SPACE), epochs=3, seed=1, amp=True, channels_last=True)


# --- engine.py ---------------------------------------------------------------


def synthetic_fitness(train_cfg) -> float:
    """Otimo em lr=1e-3 e dropout=0.2; `base_filters` alto ajuda um pouco."""
    penalty = abs(math.log10(train_cfg.optim.lr) + 3.0) / 2.0
    penalty += abs(train_cfg.model.dropout - 0.2)
    return max(0.0, 0.5 - penalty) + train_cfg.model.base_filters / 1000.0


class FakeTrainer:
    """Mock de `train`: fitness sintetico determinístico a partir do genoma.

    Otimo em lr=1e-3, dropout=0.2, base_filters alto. Registra cada chamada.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, train_cfg, data, ctx, on_event, cancel, device, keep_weights=False) -> TrainResult:
        self.calls.append(
            {
                "cfg": train_cfg,
                "ctx": ctx,
                "keep_weights": keep_weights,
                "seed": train_cfg.seed,
            }
        )
        fitness = synthetic_fitness(train_cfg)

        return TrainResult(
            best_val_acc=fitness,
            # O retreino final roda muito mais epocas, entao chega mais longe.
            # O offset deixa os dois numeros distinguiveis nos testes.
            final_val_acc=fitness + (0.1 if keep_weights else 0.0),
            final_train_loss=1.0,
            epochs_run=train_cfg.epochs,
            elapsed_s=0.0,
            cancelled=cancel.is_set(),
            state_dict={"w": torch.zeros(1)} if keep_weights else None,
        )


class Collector:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)

    def of(self, kind) -> list:
        return [e for e in self.events if isinstance(e, kind)]


@pytest.fixture
def fake_train(monkeypatch: pytest.MonkeyPatch) -> FakeTrainer:
    trainer = FakeTrainer()
    monkeypatch.setattr(engine_mod, "train", trainer)
    return trainer


def make_job(**kwargs) -> GAJob:
    base = {
        "population_size": 6,
        "generations": 4,
        "epochs_per_individual": 2,
        "elitism": 1,
        "tournament_size": 3,
        "seed": 42,
    }
    base.update(kwargs)
    return GAJob(ga=GAConfig(**base))


def go(job: GAJob, events: Collector, cancel: threading.Event | None = None) -> GAResult:
    return run_ga(
        job,
        data=None,
        run_id="run-1",
        on_event=events,
        cancel=cancel or threading.Event(),
        device=torch.device("cpu"),
    )


def test_budget_matches_the_real_train_call_count(fake_train: FakeTrainer) -> None:
    job = make_job()
    trainings, epochs = job.ga.budget()
    events = Collector()

    result = go(job, events)

    search_calls = [c for c in fake_train.calls if not c["keep_weights"]]
    assert len(search_calls) == trainings
    assert result.total_trainings == trainings
    assert sum(c["cfg"].epochs for c in search_calls) == epochs


def test_elites_are_not_retrained(fake_train: FakeTrainer) -> None:
    job = make_job(population_size=6, generations=4, elitism=2)
    events = Collector()

    go(job, events)

    # Sem elitismo seriam population_size * generations avaliacoes.
    naive = job.ga.population_size * job.ga.generations
    assert len(events.of(IndividualDone)) == job.ga.budget()[0] < naive


def test_elite_genome_survives_into_the_next_generation(
    fake_train: FakeTrainer,
) -> None:
    events = Collector()
    go(make_job(elitism=1), events)

    gens = events.of(GenerationDone)
    for previous, current in zip(gens, gens[1:]):
        assert current.best_fitness >= previous.best_fitness


def test_best_fitness_is_monotonically_non_decreasing(fake_train: FakeTrainer) -> None:
    events = Collector()
    result = go(make_job(generations=6, elitism=1), events)

    scores = [e.best_fitness for e in events.of(GenerationDone)]
    assert scores == sorted(scores)
    assert result.best_fitness == max(scores)


def test_one_generationdone_per_generation(fake_train: FakeTrainer) -> None:
    job = make_job(generations=5)
    events = Collector()

    result = go(job, events)

    done = events.of(GenerationDone)
    assert len(done) == job.ga.generations
    assert [e.gen for e in done] == list(range(job.ga.generations))
    assert result.generations_run == job.ga.generations


def test_generation_stats_describe_the_population(fake_train: FakeTrainer) -> None:
    events = Collector()
    go(make_job(generations=2, population_size=6, elitism=1), events)

    first = events.of(GenerationDone)[0]
    gen0 = [e.fitness for e in events.of(IndividualDone) if e.gen == 0]
    assert first.best_fitness == pytest.approx(max(gen0))
    assert first.mean_fitness == pytest.approx(sum(gen0) / len(gen0))
    assert first.std_fitness >= 0.0
    assert first.best_genome["lr"] > 0.0


def test_same_seed_produces_the_same_genome_sequence(fake_train: FakeTrainer) -> None:
    def genomes() -> list[dict]:
        trainer = FakeTrainer()
        events = Collector()
        run_ga(
            make_job(),
            data=None,
            run_id="r",
            on_event=events,
            cancel=threading.Event(),
            device=torch.device("cpu"),
        )
        return [e.genome for e in events.of(IndividualDone)]

    assert genomes() == genomes()


def test_different_seed_produces_a_different_sequence(
    fake_train: FakeTrainer,
) -> None:
    def genomes(seed: int) -> list[dict]:
        events = Collector()
        go(make_job(seed=seed), events)
        return [e.genome for e in events.of(IndividualDone)]

    assert genomes(42) != genomes(7)


def test_result_is_immune_to_the_global_random(fake_train: FakeTrainer) -> None:
    def genomes() -> list[dict]:
        events = Collector()
        go(make_job(), events)
        return [e.genome for e in events.of(IndividualDone)]

    random.seed(0)
    first = genomes()
    random.seed(12345)
    [random.random() for _ in range(50)]
    assert genomes() == first


def test_every_individual_uses_the_same_base_seed(fake_train: FakeTrainer) -> None:
    job = make_job(seed=1234)
    go(job, Collector())
    assert {c["seed"] for c in fake_train.calls} == {1234}


def test_runstarted_declares_the_budget(fake_train: FakeTrainer) -> None:
    job = make_job()
    events = Collector()

    go(job, events)

    started = events.of(RunStarted)
    assert len(started) == 1
    assert started[0].unit_total == job.ga.budget()[0]
    assert started[0].epochs_per_unit == job.ga.epochs_per_individual
    assert started[0].kind == "ga"


def test_unit_index_walks_the_budget_once(fake_train: FakeTrainer) -> None:
    job = make_job()
    trainings = job.ga.budget()[0]
    events = Collector()

    go(job, events)

    search = [c["ctx"] for c in fake_train.calls if not c["keep_weights"]]
    assert [c.unit_index for c in search] == list(range(trainings))
    assert {c.unit_total for c in search} == {trainings}


def test_individualdone_carries_gen_and_index(fake_train: FakeTrainer) -> None:
    events = Collector()
    go(make_job(population_size=6, generations=2, elitism=1), events)

    first_gen = [e for e in events.of(IndividualDone) if e.gen == 0]
    assert [e.individual for e in first_gen] == [0, 1, 2, 3, 4, 5]
    assert all(e.epochs == 2 for e in first_gen)
    assert all(e.run_id == "run-1" for e in first_gen)


# --- retreino final ----------------------------------------------------------


def test_state_dict_comes_from_a_single_final_retrain(
    fake_train: FakeTrainer,
) -> None:
    events = Collector()
    result = go(make_job(), events)

    keep = [c for c in fake_train.calls if c["keep_weights"]]
    assert len(keep) == 1
    assert keep[0] is fake_train.calls[-1]  # e a ultima chamada de todas
    assert result.best_state_dict is not None


def test_final_retrain_uses_the_best_genome(fake_train: FakeTrainer) -> None:
    events = Collector()
    result = go(make_job(), events)

    cfg = fake_train.calls[-1]["cfg"]
    assert cfg.model.base_filters == result.best_genome["base_filters"]
    assert cfg.optim.lr == pytest.approx(result.best_genome["lr"])
    assert cfg.model.dropout == pytest.approx(result.best_genome["dropout"])


def test_final_retrain_runs_for_final_epochs_not_the_search_budget(
    fake_train: FakeTrainer,
) -> None:
    job = make_job(epochs_per_individual=2, final_epochs=25)
    go(job, Collector())

    search = [c for c in fake_train.calls if not c["keep_weights"]]
    retrain = [c for c in fake_train.calls if c["keep_weights"]]

    assert {c["cfg"].epochs for c in search} == {2}
    assert [c["cfg"].epochs for c in retrain] == [25]


def test_final_epochs_does_not_leak_into_budget(fake_train: FakeTrainer) -> None:
    """`budget()` descreve a busca. O retreino final e extra, de proposito."""
    short = make_job(final_epochs=1)
    long = make_job(final_epochs=200)
    assert short.ga.budget() == long.ga.budget()


def test_final_val_acc_describes_the_saved_weights_not_the_search(
    fake_train: FakeTrainer,
) -> None:
    """Quem rotula o `best.pt` usa este campo, nao `best_fitness`."""
    events = Collector()
    result = go(make_job(), events)

    assert result.final_val_acc is not None
    assert result.final_val_acc > result.best_fitness
    retrain_cfg = fake_train.calls[-1]["cfg"]
    assert result.final_val_acc == pytest.approx(synthetic_fitness(retrain_cfg) + 0.1)
    assert result.best_fitness == pytest.approx(synthetic_fitness(retrain_cfg))


def test_cancelled_run_has_no_final_val_acc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(generations=3)
    cancel = threading.Event()
    trainer = FakeTrainer()

    def cancelling(*args, **kwargs) -> TrainResult:
        result = trainer(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(engine_mod, "train", cancelling)
    result = run_ga(
        job,
        data=None,
        run_id="r",
        on_event=Collector(),
        cancel=cancel,
        device=torch.device("cpu"),
    )

    assert result.final_val_acc is None
    assert result.best_state_dict is None


def test_final_retrain_does_not_overflow_the_progress_bar(
    fake_train: FakeTrainer,
) -> None:
    job = make_job()
    trainings = job.ga.budget()[0]
    go(job, Collector())

    ctx = fake_train.calls[-1]["ctx"]
    assert ctx.unit_index == trainings - 1
    assert ctx.unit_total == trainings


def test_runfinished_reports_the_search_not_the_retrain(
    fake_train: FakeTrainer,
) -> None:
    job = make_job()
    trainings, epochs = job.ga.budget()
    events = Collector()

    result = go(job, events)

    finished = events.of(RunFinished)
    assert len(finished) == 1
    assert finished[0].total_trainings == trainings
    assert finished[0].total_epochs == epochs
    assert finished[0].best_fitness == result.best_fitness
    assert finished[0].artifact == "best.pt"
    assert finished[0].kind == "ga"


# --- cancelamento ------------------------------------------------------------


def test_cancel_mid_generation_keeps_earlier_generations_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(population_size=6, generations=5, elitism=1)
    cancel = threading.Event()
    trainer = FakeTrainer()
    events = Collector()

    def cancelling(*args, **kwargs) -> TrainResult:
        result = trainer(*args, **kwargs)
        # Cancela no meio da geracao 2 (a terceira).
        if len(trainer.calls) == job.ga.population_size + 5 + 2:
            cancel.set()
        return result

    monkeypatch.setattr(engine_mod, "train", cancelling)
    result = run_ga(
        job,
        data=None,
        run_id="r",
        on_event=events,
        cancel=cancel,
        device=torch.device("cpu"),
    )

    assert result.cancelled is True
    assert result.generations_run == 2
    assert [e.gen for e in events.of(GenerationDone)] == [0, 1]
    assert events.of(RunFinished) == []
    assert result.best_state_dict is None


def test_cancel_returns_the_best_found_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(generations=5)
    cancel = threading.Event()
    trainer = FakeTrainer()
    events = Collector()

    def cancelling(*args, **kwargs) -> TrainResult:
        result = trainer(*args, **kwargs)
        if len(trainer.calls) == job.ga.population_size + 3:
            cancel.set()
        return result

    monkeypatch.setattr(engine_mod, "train", cancelling)
    result = run_ga(
        job,
        data=None,
        run_id="r",
        on_event=events,
        cancel=cancel,
        device=torch.device("cpu"),
    )

    assert result.best_fitness == max(e.best_fitness for e in events.of(GenerationDone))
    assert result.best_genome != {}


def test_cancel_reports_total_trainings_as_the_at_unit_for_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O runner emite `RunCancelled(run_id, at_unit)` com este numero.

    `total_trainings` conta so treinos concluidos, entao e exatamente o indice
    da unidade em que a run parou. Ver docs/DECISIONS.md.
    """
    job = make_job(population_size=6, generations=5, elitism=1)
    cancel = threading.Event()
    trainer = FakeTrainer()
    events = Collector()
    stop_at = 9

    def cancelling(*args, **kwargs) -> TrainResult:
        result = trainer(*args, **kwargs)
        if len(trainer.calls) == stop_at:
            cancel.set()
        return result

    monkeypatch.setattr(engine_mod, "train", cancelling)
    result = run_ga(
        job,
        data=None,
        run_id="r",
        on_event=events,
        cancel=cancel,
        device=torch.device("cpu"),
    )

    assert result.total_trainings == stop_at
    assert result.total_trainings == len(events.of(IndividualDone))
    last_ctx = trainer.calls[-1]["ctx"]
    assert last_ctx.unit_index == stop_at - 1


def test_cancel_does_not_run_the_final_retrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(generations=3)
    cancel = threading.Event()
    trainer = FakeTrainer()

    def cancelling(*args, **kwargs) -> TrainResult:
        result = trainer(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(engine_mod, "train", cancelling)
    run_ga(
        job,
        data=None,
        run_id="r",
        on_event=Collector(),
        cancel=cancel,
        device=torch.device("cpu"),
    )

    assert all(not c["keep_weights"] for c in trainer.calls)


# --- torneio -----------------------------------------------------------------


def test_tournament_prefers_higher_fitness() -> None:
    population = [engine_mod._Individual({"i": i}, fitness=float(i)) for i in range(10)]
    rng = random.Random(0)
    picks = [engine_mod._tournament(population, 5, rng).fitness for _ in range(300)]
    assert sum(picks) / len(picks) > 6.0  # bem acima da media 4.5


def test_tournament_of_size_one_is_uniform() -> None:
    population = [engine_mod._Individual({"i": i}, fitness=float(i)) for i in range(10)]
    rng = random.Random(0)
    picks = {engine_mod._tournament(population, 1, rng).fitness for _ in range(300)}
    assert len(picks) == 10


def test_tournament_uses_only_the_injected_rng() -> None:
    population = [engine_mod._Individual({"i": i}, fitness=float(i)) for i in range(10)]
    random.seed(0)
    a = [engine_mod._tournament(population, 3, random.Random(9)).fitness for _ in range(20)]
    random.seed(999)
    b = [engine_mod._tournament(population, 3, random.Random(9)).fitness for _ in range(20)]
    assert a == b


def test_unevaluated_individual_ranks_below_any_score() -> None:
    """Guarda defensiva: na pratica o torneio so ve populacao ja avaliada."""
    fresh = engine_mod._Individual({"i": 0}, fitness=None)
    scored = engine_mod._Individual({"i": 1}, fitness=0.0)

    assert engine_mod._fitness_of(fresh) == float("-inf")
    assert engine_mod._fitness_of(fresh) < engine_mod._fitness_of(scored)

    # Sempre que o avaliado entra no torneio, ele ganha.
    rng = random.Random(0)
    losses = sum(engine_mod._tournament([fresh, scored], 8, rng) is fresh for _ in range(200))
    assert losses == 0
