# Spec — tui/

Framework: Textual. Ler `docs/01-ARCHITECTURE.md` antes, especialmente o modelo
de concorrencia.

## Layout

Referencia visual: nav horizontal no topo, sidebar contextual a esquerda,
area principal ao centro. A referencia e de **layout apenas** — nao replicar
conteudo de outro projeto.

```
+--------------------------------------------------------------+
|  CNN LAB TCC                                                 |
+--------------------------------------------------------------+
|  Treinar  |  Historico  |  Usar rede  |  Config              |
+-------------------+------------------------------------------+
|                   |                                          |
|  SIDEBAR          |          AREA PRINCIPAL                  |
|  (muda por aba)   |                                          |
|                   |                                          |
+-------------------+------------------------------------------+
|  status: ocioso | fila: 0 | GPU: RTX 5070                    |
+--------------------------------------------------------------+
```

## Telas

### 1. Treinar

Sidebar com dois modos, alternados por `RadioSet`:

- **Classico**: campos para `lr`, `batch_size`, `optimizer`, `dropout`,
  `weight_decay`, `base_filters`, `n_blocks`, `epochs`, `seed`.
- **AG**: campos para `population_size`, `generations`,
  `epochs_per_individual`, `crossover_prob`, `mutation_prob`, `elitism`,
  `tournament_size`, `seed`.

Abaixo dos campos, um rotulo de orcamento vivo, recalculado a cada mudanca:
`43 treinos, 129 epocas`. Vem de `GAConfig.budget()`.

Botoes: `Iniciar`, `Parar`, `Adicionar a fila`.
Modo Sleep: um `Input` numerico "N sessoes" e um botao `Enfileirar N`.
Cada sessao enfileirada recebe uma seed diferente (`seed + i`).

Area principal durante execucao:
- barra de progresso global, alimentada por `unit_index / unit_total`
- barra de epoca, alimentada por `epoch / epochs_total`
- linha de status: `gen 2/6 · ind 5/8 · epoca 1/3`
- ultimas metricas: `train_loss`, `val_loss`, `val_acc`
- um `RichLog` com as ultimas linhas de evento
- um sparkline de `best_fitness` por geracao, no modo AG

### 2. Historico

`DataTable` com as colunas: data, tipo, status, melhor fitness, duracao, nota.
Filtro por tipo. Selecionar uma linha abre um detalhe com a config usada, a
curva de fitness por geracao e o melhor genoma.

Dados vindos de `storage.reader.list_runs()`. Sem cache: reler a cada entrada
na tela.

### 3. Usar rede

- selecionar uma run concluida (dropdown alimentado por `list_runs`)
- escolher origem da imagem: amostra do dataset ou arquivo local
- botao classificar, exibindo top-5 com confianca
- historico das predicoes, de `predictions.jsonl`

Imagens do usuario passam por conversao: abrir com PIL, converter para RGB,
redimensionar para 32x32, normalizar com as mesmas estatisticas do treino.
Copiar o arquivo original para `uploads/`.

### 4. Config

Caminho do dataset, device, `amp` ligado/desligado, `channels_last`. Persistido
em `settings.json` na raiz. Sem temas nesta POC.

## Concorrencia

```python
class TrainingApp(App):
    @work(thread=True, exclusive=True, group="runner")
    def start_runner(self) -> None:
        runner = Runner(self.queue, self.post_message, self.device)
        runner.run_forever(self._stop, self._cancel_current)
```

O callback passado ao `Runner` faz `self.post_message(...)`. Textual garante que
`post_message` a partir de outra thread e seguro. Os handlers `on_*` rodam na
UI thread e so ali tocam widget.

Envolver cada evento num `Message` do Textual:

```python
class CoreEvent(Message):
    def __init__(self, evt: Event) -> None:
        self.evt = evt
        super().__init__()
```

## Antipadroes

- Chamar `train` ou `run_ga` de um handler de botao. Sempre pela fila.
- Ler `.jsonl` na UI thread durante execucao. O progresso vem por mensagem.
- `time.sleep` na UI thread.
- Widget referenciado dentro da worker thread.

## Aceite

- [ ] Iniciar um treino nao congela a interface
- [ ] Barra de progresso avanca coerentemente nos dois modos
- [ ] `Parar` interrompe em ate uma epoca
- [ ] Enfileirar 3 sessoes e ve-las rodar em sequencia
- [ ] Fechar e reabrir mostra as runs anteriores no historico
- [ ] Redimensionar o terminal nao quebra o layout
