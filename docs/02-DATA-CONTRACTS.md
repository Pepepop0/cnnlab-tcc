# Contratos de dados

Tudo aqui é fronteira entre módulos. Mudar qualquer coisa deste documento
quebra outro módulo. Se precisar mudar, sinalize antes.

Os arquivos `src/cnnlab/config.py` e `src/cnnlab/events.py` já vêm
implementados no kit e são a fonte da verdade. Este documento explica o porquê.

---

## Configuração

Toda config é pydantic. Serializa pra JSON com `.model_dump_json()`, carrega
com `.model_validate_json()`. É assim que o `config.json` de cada run é escrito.

### Hierarquia

```
DatasetConfig    qual dataset, split, subset de classes
ModelConfig      base_filters, n_blocks, dropout
OptimConfig      optimizer, lr, weight_decay, batch_size
TrainConfig      model + optim + epochs + amp + seed
SearchSpace      faixas de cada hiperparâmetro do AG
GAConfig         população, gerações, épocas por indivíduo, operadores, search_space
JobSpec          união discriminada: ClassicJob | GAJob
```

`JobSpec` é o que entra na fila. `kind` é o discriminador.

### Nota sobre `epochs_per_individual`

No AG, cada indivíduo é treinado por poucas épocas para caber no orçamento.
Esse número é o principal controle de custo da sessão inteira:

```
treinos totais = population_size + (generations - 1) * (population_size - elitism)
épocas totais  = treinos totais * epochs_per_individual
```

Com população 8, 6 gerações, elitismo 1, 3 épocas: 8 + 5*7 = 43 treinos, 129
épocas. Esse é o número a olhar antes de apertar start.

---

## Eventos

Dataclasses congeladas (`frozen=True`). Atravessam a fronteira de thread e são
serializadas para `.jsonl`. Nenhuma delas carrega tensor, modelo ou referência
a widget.

| Evento | Quando | Vai para |
|--------|--------|----------|
| `RunStarted` | início da run | `status.json` |
| `EpochEnd` | fim de cada época | `progress.jsonl` |
| `IndividualDone` | fim da avaliação de um indivíduo | `individuals.jsonl` |
| `GenerationDone` | fim de cada geração | `generations.jsonl` |
| `RunFinished` | fim com sucesso | `summary.json` |
| `RunFailed` | exceção não tratada | `status.json` |

`EpochEnd` carrega `gen` e `individual` como `int | None`. Em treino clássico
os dois são `None`; em AG, ambos preenchidos. É isso que permite uma barra de
progresso única servir aos dois modos.

### Cálculo de progresso

O consumidor calcula porcentagem a partir de `unit_index` e `unit_total`
presentes em `EpochEnd`. O produtor é quem sabe o total — a UI nunca deve tentar
inferir.

---

## Formato dos arquivos

### `config.json`

Dump do `JobSpec` completo, incluindo a seed resolvida. Escrito uma vez.

### `status.json`

```json
{
  "run_id": "20260901-2314_ga_a3f1",
  "kind": "ga",
  "status": "running",
  "created_at": "2026-09-01T23:14:02",
  "started_at": "2026-09-01T23:14:03",
  "finished_at": null,
  "error": null
}
```

`status` é um de: `queued`, `running`, `done`, `failed`, `cancelled`.
Escrito com write atômico a cada transição.

### `progress.jsonl`

Uma linha por época. Append puro.

```json
{"ts":"...","gen":2,"individual":5,"epoch":1,"train_loss":3.91,"val_loss":3.72,"val_acc":0.121,"elapsed_s":4.2,"unit_index":19,"unit_total":43}
```

### `individuals.jsonl`

```json
{"ts":"...","gen":2,"individual":5,"genome":{"lr":0.0031,"batch_size":128,"optimizer":"adam","dropout":0.31,"weight_decay":3.1e-5,"base_filters":32,"n_blocks":3},"fitness":0.1408,"epochs":3,"elapsed_s":12.6}
```

### `generations.jsonl`

```json
{"ts":"...","gen":2,"best_fitness":0.1408,"mean_fitness":0.1122,"std_fitness":0.0184,"best_genome":{...},"elapsed_s":94.3}
```

### `summary.json`

```json
{
  "run_id":"...","kind":"ga","status":"done",
  "best_fitness":0.1731,"best_genome":{...},
  "total_trainings":43,"total_epochs":129,
  "elapsed_s":1841.2,"artifact":"best.pt",
  "device":"NVIDIA GeForce RTX 5070","torch_version":"2.13.0+cu130"
}
```

O histórico da TUI é construído lendo `status.json` e `summary.json` de cada
pasta. Os `.jsonl` só são lidos quando o usuário abre o detalhe de uma run.

### `best.pt`

```python
torch.save(
    {
        "state_dict": model.state_dict(),
        "model_config": model_cfg.model_dump(),
        "class_names": [...],
        "val_acc": 0.1731,
        "run_id": "...",
    },
    path,
)
```

Nunca salvar o objeto do modelo com pickle direto. Só `state_dict` mais o
suficiente pra reconstruir via `build_cnn`.

### `queue.json`

```json
{"jobs":[{"id":"...","spec":{...},"status":"pending"}],"updated_at":"..."}
```

Reescrito com write atômico a cada mudança de status. É o que permite retomar
o modo Sleep depois de um crash.

### `predictions.jsonl`

```json
{"ts":"...","run_id":"...","image":"uploads/foto.png","source":"upload","predicted":"apple","confidence":0.41,"top5":[["apple",0.41],["orange",0.19]]}
```

---

## Convenção de `run_id`

```
YYYYMMDD-HHMM_{kind}_{hash4}
```

`hash4` são 4 caracteres hex aleatórios, para evitar colisão se duas runs
começarem no mesmo minuto. O `run_id` é o nome da pasta.
