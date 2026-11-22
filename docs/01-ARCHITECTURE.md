# Arquitetura

## Visão geral

Processo único. A interface roda no loop asyncio do Textual. O treino roda numa
worker thread. Comunicação da thread para a UI é por mensagem; persistência é em
arquivos, não em banco.

```
                        um processo
  ┌──────────────────────────────────────────────────────┐
  │                                                       │
  │  UI thread (asyncio)  ──enfileira──▶  Fila de jobs   │
  │        ▲                                    │         │
  │        │ post_message                       ▼         │
  │        └──────────────────────  Worker thread         │
  │                                        │              │
  │                                        ▼              │
  │                                  core headless        │
  │                                  (trainer + ga)       │
  │                                        │              │
  │                                        ▼              │
  │                                   runs/ em disco      │
  └──────────────────────────────────────────────────────┘
```

## Decisões travadas

Estas decisões estão fechadas. Quem discordar deve sinalizar, não
mudar unilateralmente.

### D1 — Processo único, não daemon separado

A alternativa (worker em processo separado, comunicando por banco) foi
considerada e rejeitada para a POC. Custo de coordenação não se paga na escala
de 3 execuções. Consequência aceita: a janela do terminal precisa ficar aberta
durante o modo Sleep.

### D2 — Sistema de arquivos, não SQLite

Escala esperada: dezenas de runs, centenas de indivíduos. Listar diretórios e
ler JSON é instantâneo nessa ordem de grandeza. A camada `storage` é uma
interface — trocar por SQLite depois é mudar um módulo, não o projeto.

### D3 — Persistência a cada indivíduo, não a cada run

Granularidade fina é o que protege contra crash de madrugada. Um crash deve
custar um indivíduo, não a sessão inteira. Arquivos de evento são `.jsonl`
append-only: nunca reescreve, então nunca corrompe.

### D4 — O core não conhece a TUI

`core/` não importa nada de `tui/`. A comunicação é um callback
`on_event: Callable[[Event], None]`. Isso mantém o core executável e testável
pelo terminal, que é como ele será depurado.

### D5 — Dataset inteiro na VRAM

CIFAR-100 são 60.000 imagens 32x32x3. Em `float16` normalizado cabe em menos de
1 GB, folgado nos 12 GB da 5070. Como o AG faz muitos treinos curtos, o overhead
do `DataLoader` (workers, coleta, transferência host→device por batch) domina o
tempo. Carregar uma vez como tensor na GPU e fatiar índices na mão remove esse
custo por completo.

Consequência: data augmentation, se existir, tem que ser feito em GPU sobre
tensores, não por `torchvision.transforms` em PIL. Para a POC, augmentation
fica limitado a flip horizontal aleatório, que é uma operação de tensor.

### D6 — Arquitetura de CNN fixa e parametrizada

Não há seleção de arquitetura. Há uma família única de CNN cujo formato é
controlado por `base_filters` e `n_blocks`. O genoma mexe nesses dois e no
resto dos hiperparâmetros de treino. Topologia livre no genoma é escopo de TCC,
não de POC.

### D7 — Seed explícita em toda run

Toda execução grava sua seed no `config.json`. Reprodutibilidade é requisito de
TCC e sai barato agora.

## Modelo de concorrência

**UI thread.** Roda o app Textual. Nunca chama treino diretamente. Nunca faz
I/O bloqueante. Recebe eventos por `post_message` e atualiza widgets.

**Worker thread.** Uma só. Criada com `@work(thread=True, exclusive=True)`.
Consome jobs da fila em sequência. Para cada job: instancia o core, passa um
callback que faz duas coisas — grava no `.jsonl` e faz `post_message` pra UI.

**Cancelamento.** Um `threading.Event` é passado ao trainer. O trainer o checa
no fim de cada época e entre indivíduos. Não há cancelamento no meio de uma
época; a granularidade é aceita.

**Regra de ouro.** Nenhum objeto de widget cruza a fronteira de thread. O que
cruza são dataclasses de evento, imutáveis.

## Layout em disco

```
runs/
  20260901-2314_ga_a3f1/
    config.json          escrito uma vez, no início
    status.json          reescrito com write atômico
    progress.jsonl       append: uma linha por época
    individuals.jsonl    append: uma linha por indivíduo avaliado
    generations.jsonl    append: uma linha por geração
    best.pt              state_dict da melhor rede
    summary.json         escrito no fim, write atômico
  20260901-2350_classic_b7c2/
    config.json
    status.json
    progress.jsonl
    best.pt
    summary.json
queue.json               fila do modo Sleep, retomável
predictions.jsonl        histórico da tela de predição
uploads/                 imagens enviadas pelo usuário
```

Write atômico: escreve em `arquivo.tmp`, depois `os.replace(tmp, final)`.
`os.replace` é atômico no Windows e no POSIX.

## Estrutura de pacotes

```
src/cnnlab/
  config.py          contratos de configuração (pydantic)
  events.py          dataclasses de evento
  core/
    data.py          CIFAR-100 -> tensores em VRAM
    model.py         build_cnn(ModelConfig) -> nn.Module
    trainer.py       train(...) -> TrainResult
    ga/
      genome.py      espaço de busca, encode/decode, amostragem
      engine.py      loop evolutivo
  storage/
    layout.py        caminhos, nomes de run
    writer.py        RunWriter: append de eventos, write atômico
    reader.py        RunReader: lista e lê runs para o histórico
  worker/
    queue.py         fila persistente
    runner.py        consome a fila, executa jobs
  tui/
    app.py
    screens/         train.py, history.py, predict.py, settings.py
    widgets/
    app.tcss
  cli.py             entrypoints
```

## Regras de dependência

```
tui      -> worker, storage, config, events
worker   -> core, storage, config, events
core     -> config, events
storage  -> config, events
config   -> (nada)
events   -> (nada)
```

Setas só apontam para baixo. `core` importando `tui` é erro de arquitetura.
