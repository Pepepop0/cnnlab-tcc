# Decisões do TCC aprovadas para implementação

**Registro de 27/09/2026, aprovado por Pedro para implementação.** O aceite formal do orientador não está registrado neste workspace; viabilidade técnica e resultados ainda dependem do piloto. O [protocolo](TCC-01-PROTOCOLO.md) é a referência científica; o plano operacional versionado no SQLite será a fonte dos valores exatos usados nas runs. Alteração posterior recebe número de emenda, autor, data, motivo e impacto/repetição.

| ID | Decisão proposta | Justificativa e consequência |
|---|---|---|
| TCC-01 | Pergunta é eficiência de CNN pronta para inferência em CIFAR-100/RTX 5070; BO fora do escopo. | O pivô muda o objeto de hiperparâmetros de treino para arquitetura + representação e custo real do engine. |
| TCC-02 | Backend único de inferência é TensorRT, com ONNX como intercâmbio; PyTorch treina. | Compara custo executado na GPU local. Compatibilidade e ganho são hipóteses do piloto, não garantias por documentação. |
| TCC-03 | Fitness vetorial `min(E,L,V,S)` no mesmo engine; sem soma ponderada. | Pareto preserva trade-offs e evita pesos arbitrários. E é erro top-1 em validação; L tempo GPU batch 1; V pico incremental; S bytes do `.plan`. |
| TCC-04 | Genoma v1 tem seis blocos fixos, 2–6 ativos, arquitetura, treino e representação; FP32/FP16/BF16/INT8, poda estruturada. FP8, INT4/FP4 e 2:4 são condicionais. | Encerra domínios e reparos antes da busca; opções condicionais entram somente após admissão no piloto, com hash de espaço. |
| TCC-05 | Quatro métodos: NSGA-II conjunto, mono só E, amostragem aleatória e sequencial arquitetura→compressão. | Controles isolam busca multiobjetivo, busca sem evolução e efeito da otimização conjunta. Mesmo espaço e mesma avaliação; sequencial acessa os mesmos tipos de representação depois da arquitetura. |
| TCC-06 | Comparação primária: 5,5 h de GPU ativa por método/seed, oito seeds pareadas, aproximadamente 176 h de buscas confirmatórias; sequencial dividido em 2,75 h + 2,75 h. | Inclui falhas, poda, QAT, build e benchmark; número de avaliações únicas é eixo secundário. É previsão revisável **antes** das runs, não limite alterável retrospectivamente. |
| TCC-07 | Split 45k/5k de treino/validação; 10k de teste oficial só após finalistas; calibração só em treino. Até 72 finalistas únicos pré-selecionados por validação. | Protege generalização e evita escolha de modelo pelo teste. Finalistas retreinam 30 épocas do zero. |
| TCC-08 | Piloto separado decide treino de 5 ou 8 épocas, espaço admitido, builder, estabilidade de L/V e referência de HV; testa viabilidade do sequencial 2,75 h + 2,75 h. | Evita decisões oportunistas após ver os resultados confirmatórios. Falha de instrumento leva a emenda, não exclusão silenciosa de objetivo. |
| TCC-09 | NSGA-II permanece mesmo com alta dimensionalidade ou V×S correlacionados; H4 é exploratória. | Trocar algoritmo/objetivo após observar desempenho invalidaria comparação. Reportar saturação e correlação como limitações. |
| TCC-10 | Arquitetura local: uma GPU/um job, avaliador isolado, SQLite para novo experimento, artefatos com hash. | Retomada, consultas e evidência de cada tentativa tornam-se centrais; JSONL da POC fica histórico. |
| TCC-11 | TUI opera treinos e alimenta a base; Streamlit visualiza, compara, exporta e permite **anotações separadas**, sem editar dado científico. | Preserva a diferença funcional decidida; leitura do resultado não reescreve a evidência. CLI usa o mesmo agendador da TUI. |
| TCC-12 | Sessões preferidas até 24 h, até 48 h em necessidade; prazo até fim de novembro; entrega inclui registros, análise e documentação. | O limite é operacional, não uma alteração do orçamento científico. Piloto e finalistas somam tempo fora das 176 h. |
| TCC-13 | Seeds exatas, ordem de runs, corte do orçamento, tolerâncias numéricas e política de interrupção ficam no SQLite, em plano versionado e congelado antes das runs confirmatórias. | O banco registra a configuração efetiva; export/hash garantem auditoria mesmo com o banco fora do Git. |

## Substituição controlada da POC

`docs/00-PRD.md`, `01-ARCHITECTURE.md`, `02-DATA-CONTRACTS.md`, specs `03`–`10` e o `AGENTS.md` atual regulam a POC. Pedro aprovou a migração, mas ainda é necessário **publicar** contratos v2 de config/eventos, novo schema, worker e persistência antes de editar os contratos compartilhados. Até lá, a proibição de alterar `src/cnnlab/config.py` e `events.py` continua válida. Nenhum documento de planejamento altera esses arquivos por si. A POC e seus resultados não provam o desempenho do TCC; podem ilustrar origem do sistema, rotulados como históricos.

## Ainda sujeito a comprovação, não a escolha livre pós-resultado

Há decisões condicionais com regra predefinida: formatos FP8/4-bit/2:4, 5 versus 8 épocas, ruído do benchmark e viabilidade da medição V. O piloto publica os resultados e fecha a configuração antes da primeira run confirmatória. O orientador pode solicitar mudança na revisão inicial; então os documentos recebem nova versão coerente. Depois do congelamento, mudança de pergunta, métrica, baseline, budget ou espaço exige emenda e avaliação explícita do que precisa ser repetido.
