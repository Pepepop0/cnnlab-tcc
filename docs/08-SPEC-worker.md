# Spec — worker/

Dois arquivos: `queue.py` e `runner.py`.

---

## queue.py

Fila persistente. E o que torna o modo Sleep retomavel.

```python
@dataclass
class Job:
    id: str
    spec: JobSpec
    status: Literal["pending", "running", "done", "failed", "cancelled"]
    run_id: str | None = None


class JobQueue:
    def load(self) -> None: ...
    def add(self, spec: JobSpec) -> Job: ...
    def add_many(self, specs: list[JobSpec]) -> list[Job]: ...
    def next_pending(self) -> Job | None: ...
    def mark(self, job_id: str, status: str, run_id: str | None = None) -> None: ...
    def pending_count(self) -> int: ...
    def clear_finished(self) -> None: ...
```

Persistida em `queue.json` com write atomico a cada mudanca. Ao carregar, um
job com status `running` significa crash: volta para `pending`.

---

## runner.py

```python
class Runner:
    def __init__(self, queue: JobQueue, on_event: Callable[[Event], None], device: torch.device) -> None: ...
    def run_forever(self, stop: threading.Event, cancel_current: threading.Event) -> None: ...
    def run_one(self, job: Job, cancel: threading.Event) -> None: ...
```

`run_forever` e o loop do modo Sleep:

```
enquanto nao stop.is_set():
    job = queue.next_pending()
    se job e None:
        stop.wait(2.0)   # nunca busy-wait
        continua
    queue.mark(job.id, "running")
    run_one(job, cancel_current)
    cancel_current.clear()   # cancelar um job nao mata a fila
```

`run_one`:

```
run_id = new_run_id(job.spec.kind)
writer = RunWriter(run_id, job.spec)
callback = lambda evt: (writer.handle(evt), on_event(evt))
try:
    data = load_data(job.spec.dataset, device)
    se kind == "classic": resultado = train(...)
    senao:                resultado = run_ga(...)
    writer.save_artifact(...)
    callback(RunFinished(...))
    queue.mark(job.id, "done", run_id)
except Exception as exc:
    writer.fail(exc)
    callback(RunFailed(run_id, str(exc), traceback.format_exc()))
    queue.mark(job.id, "failed", run_id)
```

## Regras

1. Uma excecao num job **nunca** derruba o loop. Marca como falho e segue para
   o proximo. E o requisito central do modo Sleep.
2. Dois eventos de cancelamento distintos:
   - `cancel_current`: aborta o job em execucao, a fila continua
   - `stop`: encerra o loop apos o job atual
3. `load_data` deve reaproveitar o cache entre jobs com a mesma
   `DatasetConfig`. Recarregar CIFAR-100 a cada job desperdicia dezenas de
   segundos por job durante a noite.
4. Entre jobs, chamar `torch.cuda.empty_cache()` e `gc.collect()`.

## Aceite

- [ ] Fila com 3 jobs executa os 3 em sequencia sem intervencao
- [ ] Um job que levanta excecao nao impede os seguintes
- [ ] Matar o processo com um job `running` e reabrir volta ele para `pending`
- [ ] `cancel_current` aborta o job atual e o proximo comeca normalmente
- [ ] Fila vazia nao consome CPU
