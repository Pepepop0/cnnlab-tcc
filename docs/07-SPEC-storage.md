# Spec — storage/

Tres arquivos: `layout.py`, `writer.py`, `reader.py`.

---

## layout.py

```python
RUNS_DIR = Path("runs")
UPLOADS_DIR = Path("uploads")
QUEUE_FILE = Path("queue.json")
PREDICTIONS_FILE = Path("predictions.jsonl")


def new_run_id(kind: RunKind, now: datetime | None = None) -> str: ...
def run_dir(run_id: str) -> Path: ...
def atomic_write_json(path: Path, payload: dict) -> None: ...
def append_jsonl(path: Path, payload: dict) -> None: ...
```

`new_run_id` produz `YYYYMMDD-HHMM_{kind}_{hash4}` com 4 hex aleatorios.

`atomic_write_json` escreve em `path.with_suffix('.tmp')` e faz
`os.replace(tmp, path)`. Nunca escreva JSON direto no destino final.

`append_jsonl` abre em modo `"a"`, escreve uma linha, faz `flush()`. Nao mantem
handle aberto entre chamadas: a POC nao tem volume que justifique, e handle
fechado sobrevive a crash.

---

## writer.py

```python
class RunWriter:
    def __init__(self, run_id: str, spec: JobSpec) -> None: ...
    def start(self) -> None: ...
    def handle(self, evt: Event) -> None: ...
    def save_artifact(
        self, state_dict: dict, model_cfg: ModelConfig, class_names: list[str], val_acc: float
    ) -> Path: ...
    def fail(self, exc: BaseException) -> None: ...
```

`handle` e o callback que o core recebe como `on_event`. Roteamento:

| Evento | Acao |
|--------|------|
| `RunStarted` | grava `status.json` com `running` |
| `EpochEnd` | append em `progress.jsonl` |
| `IndividualDone` | append em `individuals.jsonl` |
| `GenerationDone` | append em `generations.jsonl` |
| `RunFinished` | grava `summary.json` e `status.json` com `done` |
| `RunFailed` | grava `status.json` com `failed` e a mensagem |
| `RunCancelled` | grava `status.json` com `cancelled` |

`__init__` cria a pasta e grava `config.json` imediatamente. Se o processo
morrer no segundo seguinte, a run ja existe no disco com sua config.

`save_artifact` usa o formato descrito em `docs/02-DATA-CONTRACTS.md`.

---

## reader.py

```python
@dataclass
class RunSummary:
    run_id: str
    kind: RunKind
    status: RunStatus
    created_at: str
    best_fitness: float | None
    elapsed_s: float | None
    note: str


def list_runs(kind: RunKind | None = None) -> list[RunSummary]: ...
def read_progress(run_id: str) -> list[dict]: ...
def read_generations(run_id: str) -> list[dict]: ...
def read_individuals(run_id: str) -> list[dict]: ...
def read_config(run_id: str) -> JobSpec: ...
def artifact_path(run_id: str) -> Path | None: ...
```

`list_runs` le apenas `status.json` e `summary.json` de cada pasta. Nao toca nos
`.jsonl` — e isso que mantem a tela de historico instantanea. Ordena por
`created_at` decrescente.

Uma run cuja pasta existe mas cujo `status.json` diz `running` e o processo nao
esta rodando e uma run orfa (crash). `list_runs` deve marca-la como `failed` na
leitura, sem reescrever o disco, e a UI mostra assim.

Leitura de `.jsonl` deve tolerar a ultima linha truncada: envolva o
`json.loads` de cada linha num try e ignore linhas invalidas. Isso acontece
naturalmente se o processo morrer no meio de um write.

## Aceite

- [ ] `RunWriter` cria a pasta e `config.json` antes de qualquer treino
- [ ] Matar o processo no meio de uma run deixa `progress.jsonl` legivel
- [ ] `list_runs` com 50 runs retorna em menos de 100 ms
- [ ] `list_runs(kind="ga")` filtra corretamente
- [ ] Um `.jsonl` com ultima linha truncada nao levanta excecao na leitura
