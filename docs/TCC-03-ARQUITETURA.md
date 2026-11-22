# Arquitetura-alvo do sistema de pesquisa

**Proposta 1.0 · 27/09/2026 · aprovada por Pedro para implementação; aceite formal do orientador não registrado.** [Índice](TCC-00-LEIA-ME.md) · [mapa geral Archify](arquitetura-tcc.html) · [ciclo detalhado do AG](fluxo-ag-tcc.html) · [explicação passo a passo](TCC-08-FLUXO-DETALHADO.md). Os JSONs de origem dos diagramas são [mapa geral](arquitetura-tcc.archify.json) e [ciclo do AG](fluxo-ag-tcc.archify.json).

## Limite e diferença em relação à POC

A POC atual é um processo único, uma thread de treino, objetivo único e persistência em JSON/JSONL. O TCC proposto preserva o core independente da interface e a operação local, mas muda o objeto de estudo para o **engine executado** e acrescenta busca multiobjetivo, exportação/compilação TensorRT, benchmark isolado, persistência relacional e análise somente de leitura. Este documento é a **arquitetura-alvo**, não uma descrição do que já foi implementado. As decisões D1–D7 em `01-ARCHITECTURE.md` eram travas da POC; não são contratos aplicáveis automaticamente ao TCC. A troca deve ser aprovada e publicada como contratos v2 antes de alterar código, inclusive `config.py` e `events.py`.

## Componentes e fronteiras

| Componente | Responsabilidade | Não deve fazer |
|---|---|---|
| TUI Textual | Criar e acompanhar jobs de treino/busca, fila, progresso, cancelamento e retomada. | Calcular fitness, benchmark ou editar resultados. |
| CLI headless | Operar o mesmo agendador sem janela; útil para sessões longas. | Duplicar lógica científica da TUI. |
| Agendador/worker | Consumir fila durável, reservar a GPU, alternar métodos/blocos, contabilizar tempo ativo, manter heartbeat e estados. | Decidir seleção genética ou alterar medidas. |
| Core de pesquisa | Genoma/reparo, quatro métodos de busca, treino e seleção, arquivo Pareto; emite eventos tipados. | Importar TUI, Streamlit ou widgets. |
| Pipeline de modelo | Poda física, ajuste, quantização Q/DQ, exportação ONNX, compilação, verificação de engine. | Reportar custo de um checkpoint como se fosse custo do engine. |
| Avaliador isolado | Carregar um engine por vez, conferir precisão, medir E/L/V/S e diagnósticos de ruído. | Compartilhar dataset inteiro ou estado CUDA do treino com o benchmark. |
| Storage v2 | Transações SQLite, registro de tentativas/medidas, hash e catálogo de artefatos; interface de leitura estável. | Sobrescrever medidas ou selecionar “a melhor” silenciosamente. |
| Streamlit | Ler banco e artefatos, comparar métodos, filtrar, exportar tabelas/figuras; anotar interpretações em tabelas separadas. | Alterar genomas, seeds, fitness, runs ou agendar treino. |

Fluxo lógico: `TUI/CLI → fila durável → worker → core → pipeline → avaliador → storage → Streamlit`. O CIFAR-100 fornece exclusivamente treino/validação à busca; o teste oficial é aberto por uma etapa separada após congelar finalistas. Artefatos (`.pt`, `.onnx`, `.plan`, amostras de benchmark) ficam em diretórios por run, com hash no catálogo. **SQLite é a fonte de verdade do novo experimento**; JSONL da POC continua histórico importável, não recebe escrita dupla operacional.

## Modelo de execução

Uma RTX 5070, um job de GPU por vez. A TUI fica no loop da interface e recebe eventos serializáveis via passagem thread-safe; jamais manipular widget na worker thread. O worker pode orquestrar subprocessos para treino/compilação e deve executar **benchmark em processo separado**, depois de desalocar treino e dataset da VRAM. O processo de benchmark inicializa seu próprio contexto CUDA, mede o baseline de memória, carrega o `.plan`, executa e termina; isso evita que o estado de treino contamine V. O core recebe callback/event sink e não conhece a interface.

Fila e ciclo de vida duráveis: `queued → running → done|failed|cancelled`; `running` sem heartbeat após queda vira `interrupted`, não `failed` por inferência. Retomar significa reconstruir o estado do algoritmo a partir de checkpoint RNG/população/arquivo/contador e do último indivíduo **confirmado** no banco; uma tentativa interrompida pode ser repetida com novo `attempt_id`, preservando o custo da anterior. Repetição da mesma candidata/contexto usa chave idempotente; não duplica medida confirmada. Cancelamento é cooperativo entre etapas/épocas e deixa artefatos/estado íntegros. O orçamento de GPU é acumulado em segmentos monotônicos por job, com pausas excluídas e tentativas falhas incluídas.

Um único escritor transacional serializa mudanças no banco; Streamlit abre conexão somente leitura, sob WAL, sem bloquear o trabalho prolongado. A capacidade de leitura simultânea e a política de checkpoints WAL precisam de teste de carga local. Os arquivos grandes não entram em BLOB SQLite. Escrever artefato em temporário, verificar hash, publicar com substituição atômica e só então confirmar no banco. Um recovery scan encontra arquivos órfãos e registros cujo arquivo falte, sem fabricar um sucesso experimental.

## Contratos que precisam de v2 antes da implementação

1. `JobSpec`: `protocol_version`, `method`, `seed_block`, `space_hash`, `budget_gpu_s`, split e modos `pilot|confirmatory|finalist`; configuração imutável depois de enfileirar.
2. `Genome`: bruto, expresso, reparos, contexto de cache e versão do espaço; operadores e validade em [genoma](TCC-02-GENOMA.md).
3. `Evaluation`: ID de candidata e tentativa, estágio, status/código de falha, duração por etapa, hashes de engine/artefatos, fitness e incerteza/diagnóstico; unidades explicitadas no [protocolo](TCC-01-PROTOCOLO.md).
4. `Event`: início/fim de job/etapa/indivíduo, medição, exceção, heartbeat e status; objetos pequenos, imutáveis e sem tensores.
5. `Reader`: consultas versionadas de runs, pontos de frente e séries temporais; Streamlit não consulta tabelas de escrita diretamente para fazer suposições de versão.

As dependências continuam unidirecionais: interfaces → aplicação/agendador → core/pipeline/avaliador → contratos/infra. Storage implementa porta do core; o core não importa `sqlite3`. A interface de treino e a de análise são deliberadamente distintas. A CLI e a TUI chamam o mesmo serviço; Streamlit nunca chama o worker.

## Critérios de aceitação arquiteturais

- Uma busca headless e uma via TUI geram o mesmo tipo de registro; o Streamlit mostra ambos sem converter dados manualmente.
- Crash forçado após uma avaliação confirmada perde no máximo a tentativa em andamento; retomar preserva seeds, tempo gasto, cache e arquivo Pareto.
- Execução de benchmark duas vezes do mesmo engine cria duas tentativas auditáveis, sem sobrescrever a primeira.
- Uma medição inválida não aparece como ponto da frente nem como zero; continua contada nas tentativas e horas.
- Teste oficial não é acessível por caminhos de busca/piloto; só pelo job de finalistas congelados.
- SQLite, artefatos e splits têm backup/exportação testada antes da primeira run confirmatória.

## Riscos conhecidos

TensorRT/ONNX/QAT/poda podem não viabilizar combinações do espaço na GPU local; o teste de admissão decide o espaço definitivo. Engines `.plan` são dependentes do ambiente; o manifesto precisa permitir reproduzir o build, não prometer portabilidade binária. A medição de pico V e o controle térmico são riscos científicos, não só de engenharia. Falha em qualquer um suspende confirmação, em vez de trocar objetivo ou instrumento sem emenda documentada.
