"""Contratos de configuracao.

Este modulo e fronteira entre todos os outros. Nao importa nada do projeto.
Alterar qualquer campo aqui quebra outros modulos: sinalize antes de mudar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Optimizer = Literal["adam", "sgd_momentum"]


class DatasetConfig(BaseModel):
    name: Literal["cifar100"] = "cifar100"
    root: Path = Path("./data")
    val_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    # None = as 100 classes. Um inteiro N usa apenas as N primeiras classes,
    # util para acelerar a demonstracao.
    class_subset: int | None = Field(default=None, ge=2, le=100)
    # Carrega o dataset inteiro na VRAM como tensor. Ver decisao D5.
    cache_on_device: bool = True
    augment_flip: bool = True


class ModelConfig(BaseModel):
    base_filters: int = Field(default=32, ge=8, le=128)
    n_blocks: int = Field(default=3, ge=2, le=4)
    dropout: float = Field(default=0.25, ge=0.0, le=0.7)


class OptimConfig(BaseModel):
    optimizer: Optimizer = "adam"
    lr: float = Field(default=1e-3, gt=0.0, lt=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, lt=1.0)
    batch_size: int = Field(default=128, ge=16, le=512)


class TrainConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    epochs: int = Field(default=10, ge=1, le=200)
    amp: bool = True
    channels_last: bool = True
    seed: int = 42


class SearchSpace(BaseModel):
    """Espaco de busca do AG. Faixas inclusivas."""

    lr_log10: tuple[float, float] = (-4.0, -2.0)
    weight_decay_log10: tuple[float, float] = (-6.0, -3.0)
    dropout: tuple[float, float] = (0.0, 0.5)
    batch_size: list[int] = [64, 128, 256]
    optimizer: list[Optimizer] = ["adam", "sgd_momentum"]
    base_filters: list[int] = [16, 32, 64]
    n_blocks: list[int] = [2, 3]


class GAConfig(BaseModel):
    population_size: int = Field(default=8, ge=4, le=64)
    generations: int = Field(default=6, ge=1, le=100)
    epochs_per_individual: int = Field(default=3, ge=1, le=50)
    # Epocas do retreino final do melhor genoma, que produz o `best.pt`. E um
    # treino so, nao N: pode ser bem mais longo que os da busca sem pesar no
    # orcamento. Nao entra em `budget()`, que descreve apenas a busca.
    final_epochs: int = Field(default=25, ge=1, le=200)
    crossover_prob: float = Field(default=0.6, ge=0.0, le=1.0)
    mutation_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    mutation_sigma: float = Field(default=0.15, gt=0.0, le=1.0)
    tournament_size: int = Field(default=3, ge=2, le=10)
    elitism: int = Field(default=1, ge=0, le=8)
    seed: int = 42
    search_space: SearchSpace = SearchSpace()

    def budget(self) -> tuple[int, int]:
        """Retorna (treinos_totais, epocas_totais). Chame antes de iniciar."""
        offspring = max(self.population_size - self.elitism, 0)
        trainings = self.population_size + (self.generations - 1) * offspring
        return trainings, trainings * self.epochs_per_individual


class ClassicJob(BaseModel):
    kind: Literal["classic"] = "classic"
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    note: str = ""


class GAJob(BaseModel):
    kind: Literal["ga"] = "ga"
    dataset: DatasetConfig = DatasetConfig()
    ga: GAConfig = GAConfig()
    # Campos de treino nao cobertos pelo genoma.
    amp: bool = True
    channels_last: bool = True
    note: str = ""


JobSpec = Annotated[ClassicJob | GAJob, Field(discriminator="kind")]


class JobSpecEnvelope(BaseModel):
    """Wrapper para (de)serializar um JobSpec isolado."""

    spec: JobSpec
