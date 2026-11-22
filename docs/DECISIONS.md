# Log de decisoes

Registre aqui toda decisao tomada durante a implementacao que se desvie das
specs, ou toda escolha nao coberta por elas.

Formato:

```
## AAAA-MM-DD — titulo curto
**Contexto:** o que estava sendo feito
**Problema:** o que travou
**Decisao:** o que foi feito
**Consequencia:** o que isso implica para outros modulos
```

---

## 2026-09-01 — Kit inicial

**Contexto:** definicao da arquitetura antes de comecar a implementacao.
**Decisao:** processo unico com worker thread, persistencia em sistema de
arquivos, dataset inteiro em VRAM. Detalhes e alternativas rejeitadas em
`docs/01-ARCHITECTURE.md`.
**Consequencia:** o terminal precisa ficar aberto durante o modo Sleep.

## 2026-09-01 — `note` gravado no `status.json`

**Contexto:** implementacao de `storage/`.
**Problema:** `RunSummary` tem campo `note`, mas `note` so existe no `JobSpec`,
que mora no `config.json` — e a spec proibe `list_runs` de abrir `config.json`.
**Decisao:** `RunWriter` grava `note` tambem no `status.json` (e no
`summary.json`). O reader le de la, com fallback para `""`.
**Consequencia:** `status.json` tem uma chave a mais que o exemplo do
`docs/02-DATA-CONTRACTS.md`. E superset: quem le as chaves documentadas nao
muda. Nenhum outro modulo precisa se adaptar.

## 2026-09-01 — `tui/screens/` sao containers, nao `Screen` do Textual

**Contexto:** casca da TUI, layout do `docs/09-SPEC-tui.md`.
**Problema:** a spec pede nav no topo e barra de status embaixo sempre
visiveis. Com `Screen` do Textual e `push_screen`, cada tela teria que
recompor a moldura inteira.
**Decisao:** uma unica `Screen`. `TrainView`, `HistoryView`, `PredictView` e
`SettingsView` sao `Horizontal` (sidebar + area principal) trocados por um
`ContentSwitcher`, com `Tabs` controlando qual esta ativo. Os arquivos
continuam em `tui/screens/` como manda a spec.
**Consequencia:** o wiring encontra todas as telas montadas ao mesmo tempo;
`query_one` a partir do app precisa de id, nao de tipo de tela. Nenhum
`push_screen` no caminho comum.

## 2026-09-01 — Moldura sem `dock`

**Contexto:** titulo, nav, corpo e barra de status.
**Problema:** dois widgets docados na mesma borda (`dock: top`) se sobrepoem —
o titulo sumiu atras da nav.
**Decisao:** a `Screen` e uma coluna: titulo (1 linha), nav, corpo (`1fr`) e
status (1 linha), sem `dock` nenhum.
**Consequencia:** ordem no `compose` define a moldura. Quem mexer no layout
precisa manter essa ordem.

## 2026-09-01 — Deteccao de run orfa por registro em memoria

**Contexto:** `list_runs` deve mostrar como `failed` uma run cujo `status.json`
diz `running` mas que nao esta rodando.
**Problema:** sem PID no `status.json` nao da para saber se a run e viva. Por o
PID no arquivo seria mais uma chave fora do contrato e ainda exigiria consultar
o SO.
**Decisao:** `layout` mantem um set em memoria de runs vivas. `RunWriter`
registra no `__init__` e desregistra no evento terminal. `list_runs` marca como
`failed` qualquer `running` fora desse set, sem escrever no disco.
**Consequencia:** vale pela decisao D1 (processo unico) — uma run `running` que
nao foi aberta por este processo so pode ser resto de crash. Se algum dia o
worker virar processo separado, isso precisa virar checagem de PID.

## 2026-09-01 — Torneio a mao em vez de `deap.tools.selTournament`

**Contexto:** implementacao de `core/ga/engine.py`. A spec `06-SPEC-core-ga.md`
indica DEAP como escolha padrao para selecao por torneio.
**Problema:** DEAP 1.4 instala e importa normalmente no Python 3.14 — a falha
prevista na spec nao aconteceu. Mas `deap.tools.selTournament` sorteia pelo
modulo `random` **global** e nao aceita uma instancia de `random.Random`. Usar
a funcao do DEAP quebraria a regra 5 da spec ("toda aleatoriedade do AG vem do
`random.Random(job.ga.seed)`") e, com ela, a reprodutibilidade por seed exigida
pela decisao D7.
**Decisao:** `engine._tournament` implementa torneio com reposicao em 6 linhas,
recebendo o `rng` como parametro. Mesma semantica do `selTournament` (k sorteios
com reposicao, vence o maior fitness). DEAP continua como dependencia declarada
no `pyproject.toml`, mas o `core/ga/` nao o importa.
**Consequencia:** duas runs com a mesma seed produzem a mesma sequencia de
genomas independentemente do estado do `random` global — ha teste cobrindo isso
(`test_result_is_immune_to_the_global_random`). Se algum modulo futuro quiser
estatisticas do DEAP (`tools.Statistics`), o mesmo cuidado vale: verificar se a
funcao aceita rng proprio antes de adotar.

## 2026-09-01 — `RunCancelled` e responsabilidade do runner, nao do core

**Contexto:** `core/ga/engine.run_ga` e `core/trainer.train` retornam
`cancelled=True` quando o `threading.Event` de cancelamento e setado.
**Problema:** nenhuma spec diz quem emite o evento terminal nesse caso. O
pseudocodigo de `docs/08-SPEC-worker.md` so cobre `RunFinished` e `RunFailed`;
`RunCancelled` existe em `events.py` mas nao aparece em spec nenhuma. Sem
dono definido, ou ninguem emite (a run fica `running` para sempre no
`status.json`) ou o core emite e passa a decidir o ciclo de vida da run, o que
contraria a decisao D4.
**Decisao:** o core **nunca** emite evento terminal em cancelamento — nem
`RunFinished`, nem `RunCancelled`. Retorna o resultado parcial e para. O
`worker/runner.py` e o dono: ao receber `GAResult.cancelled` ou
`TrainResult.cancelled`, emite `RunCancelled(run_id, at_unit)` como evento
terminal e chama `JobQueue.mark(job_id, "cancelled", run_id)`.

De onde sai o `at_unit`:
- AG: `GAResult.total_trainings` — conta apenas treinos concluidos, entao e
  exatamente o indice da unidade em que a run parou.
- Classico: o runner construiu o `UnitContext`, entao conhece o `unit_index`
  (0, com `unit_total=1`).

**Consequencia:** `run_ga` cancelada tambem nao faz o retreino final, entao
`GAResult.best_state_dict` vem `None` e nao ha `best.pt` para salvar. O runner
precisa tratar isso: `summary.json` de run cancelada nao tem artefato. A fila
ja suporta o status — `queue.mark` aceita `"cancelled"` e `clear_finished` o
trata como terminal.

## 2026-09-01 — A tela Usar rede importa `core/predict.py` direto

**Contexto:** wiring de `tui/screens/predict.py`.
**Problema:** `docs/01-ARCHITECTURE.md` lista `tui -> worker, storage, config,
events`. `core` nao aparece. Mas a inferencia so existe em
`core/predict.py`, e `worker/` so tem fila e runner de **jobs de treino** —
nao ha caminho pela camada worker para classificar uma imagem, e criar um
job de predicao na fila seria absurdo para uma acao interativa de 1 segundo.
**Decisao:** a tela importa `cnnlab.core.predict` e `cnnlab.core.data`
**dentro** da funcao que roda na worker thread, nunca no topo do modulo. A
regra que realmente protege a arquitetura continua valida: `core` nao sabe que
a TUI existe, e nenhum widget e tocado fora da UI thread — o resultado volta
como `Message` com uma dataclass congelada, sem tensor.
**Consequencia:** a seta `tui -> core` passa a existir, so para leitura de
predicao. Importar torch tarde tambem mantem o startup da TUI rapido e o resto
da interface testavel sem GPU. Se um dia a predicao virar tipo de job, isso
volta para o runner e o import sai daqui.

## 2026-09-01 — `predictions.jsonl` e escrito pelo core, nao pela TUI

**Contexto:** `docs/10-SPEC-predict.md` regra 5 poe o registro em
`core/predict.py`; `_record_prediction` ja grava toda predicao.
**Decisao:** a tela nao escreve nada. So le o arquivo para montar o historico,
com o mesmo cuidado do reader (linha truncada e ignorada).
**Consequencia:** predicao feita por script, sem TUI, tambem aparece no
historico. O campo `source` gravado pelo core e `"tensor"` para amostra do
dataset e `"upload"` para arquivo; a tela traduz `"tensor"` para `amostra` na
exibicao.

## 2026-09-01 — Divida: `RunFinished` do AG e emitido antes do `best.pt` existir

**Contexto:** no modo AG, `run_ga` emite `RunFinished(artifact="best.pt")` de
dentro do core; quem grava o arquivo e o `Runner`, depois que `run_ga` retorna.
**Problema:** ha uma janela de milissegundos em que o `summary.json` ja promete
um artefato que ainda nao esta no disco. Um crash exatamente ali deixa a run
com `status=done`, `artifact="best.pt"` e nenhum `best.pt`.
**Decisao:** nao corrigir agora. Corrigir exigiria mover a emissao do
`RunFinished` para fora do engine ou dar ao engine acesso ao `storage`, e o
segundo quebra a regra de dependencia (`core` nunca importa `storage`). No modo
classico o problema nao existe: o runner grava o artefato antes de emitir.
**Consequencia:** quem consome artefato usa `reader.artifact_path`, que checa o
arquivo no disco em vez de confiar no `summary.json` — ele devolve `None` nesse
caso e a tela de predicao ja filtra runs sem `best.pt`. Sintoma benigno. Se um
dia o engine parar de emitir seu proprio `RunFinished`, essa divida some junto.
