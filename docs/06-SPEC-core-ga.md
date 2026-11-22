# Spec — core/ga/

Dois arquivos: `genome.py` (representacao) e `engine.py` (loop evolutivo).

---

## genome.py

### Interface

```python
Genome = dict[str, Any]
# chaves: lr, weight_decay, dropout, batch_size, optimizer, base_filters, n_blocks


def sample(space: SearchSpace, rng: random.Random) -> Genome: ...
def to_train_config(g: Genome, epochs: int, seed: int, amp: bool, channels_last: bool) -> TrainConfig: ...
def crossover(a: Genome, b: Genome, rng: random.Random) -> tuple[Genome, Genome]: ...
def mutate(g: Genome, space: SearchSpace, sigma: float, rng: random.Random) -> Genome: ...
def clamp(g: Genome, space: SearchSpace) -> Genome: ...
```

### Representacao

Genoma e um dicionario, nao uma lista. Legibilidade no `.jsonl` vale mais que
elegancia de encoding aqui.

Parametros continuos sao armazenados no espaco em que sao amostrados:
- `lr` e `weight_decay` vivem em log10 internamente, mas o dicionario guarda o
  valor linear ja convertido. A mutacao converte para log10, perturba, volta.
- `dropout` e linear.
- `batch_size`, `optimizer`, `base_filters`, `n_blocks` sao categoricos.

### Operadores

- **crossover**: uniforme, por chave, probabilidade 0.5 de trocar cada gene.
- **mutate**: para cada chave, com probabilidade independente:
  - continuo: gaussiana com desvio `sigma * (max - min)` no espaco de amostragem,
    depois clamp na faixa.
  - categorico: escolhe outro valor da lista, uniformemente.
- **clamp**: garante que todo genoma esta dentro de `SearchSpace`. Chamar sempre
  depois de crossover e mutate.

---

## engine.py

### Interface

```python
@dataclass
class GAResult:
    best_genome: Genome
    best_fitness: float
    best_state_dict: dict | None
    generations_run: int
    total_trainings: int
    cancelled: bool


def run_ga(
    job: GAJob,
    data: DataBundle,
    run_id: str,
    on_event: Callable[[Event], None],
    cancel: threading.Event,
    device: torch.device,
) -> GAResult: ...
```

### Loop

```
rng = random.Random(job.ga.seed)
trainings, _ = job.ga.budget()
emitir RunStarted(unit_total=trainings, epochs_per_unit=epochs_per_individual)

pop = [sample(space, rng) for _ in range(population_size)]
unit = 0

para gen em 0..generations-1:
    para i, genome em enumerate(pop):
        se ja avaliado (elite carregada), pula e reaproveita fitness
        cfg = to_train_config(genome, epochs_per_individual, seed, ...)
        ctx = UnitContext(run_id, unit, trainings, gen, i)
        res = train(cfg, data, ctx, on_event, cancel, device,
                    keep_weights=False)
        fitness = res.best_val_acc
        emitir IndividualDone
        unit += 1
        se cancel setado: retornar GAResult(cancelled=True) com o melhor ate agora

    emitir GenerationDone(best, mean, std, best_genome)

    se gen == generations-1: parar
    elites = top `elitism` por fitness (mantem genoma e fitness)
    filhos = []
    enquanto len(filhos) < population_size - elitism:
        p1 = tournament(pop, tournament_size, rng)
        p2 = tournament(pop, tournament_size, rng)
        se rng.random() < crossover_prob: c1, c2 = crossover(p1, p2, rng)
        senao: c1, c2 = copia(p1), copia(p2)
        para c em (c1, c2):
            se rng.random() < mutation_prob: c = mutate(c, space, sigma, rng)
            filhos.append(clamp(c, space))
    pop = elites + filhos[:population_size - elitism]

retreinar o melhor genoma com keep_weights=True para obter o state_dict final
emitir RunFinished
```

### Regras

1. Fitness e `best_val_acc` do treino curto. Maior e melhor.
2. Elitismo carrega genoma **e** fitness: elites nao sao retreinadas. E o que
   torna a conta de `budget()` correta.
3. O `state_dict` da melhor rede sai de **um retreino final** do melhor genoma,
   nao de manter pesos durante a busca. Isso mantem a VRAM previsivel.
4. Cada individuo usa a mesma seed base. Variacao entre individuos vem do
   genoma, nao de aleatoriedade de inicializacao. Isso reduz ruido no fitness,
   que e critico com poucas epocas.
5. Toda aleatoriedade do AG vem do `random.Random(job.ga.seed)`. Nada de
   `random` global.

### DEAP

DEAP e a escolha padrao para selecao por torneio e estatisticas. Se a instalacao
falhar no Python 3.14, implemente `tournament` a mao — sao 8 linhas — e siga.
Nao bloqueie o projeto por causa disso. Registre a decisao em
`docs/DECISIONS.md`.

### Aceite

- [ ] `budget()` bate com a contagem real de chamadas a `train`
- [ ] Mesma seed produz a mesma sequencia de genomas
- [ ] `generations.jsonl` tem exatamente `generations` linhas
- [ ] `best_fitness` e monotonicamente nao decrescente entre geracoes (com elitismo >= 1)
- [ ] Cancelar no meio da geracao 3 grava as geracoes 0 a 2 intactas
