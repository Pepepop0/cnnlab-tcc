# Fases e estágios de implementação

**Visão deliberadamente de alto nível · 27/09/2026.** Plano aprovado por Pedro para iniciar implementação; o piloto e o congelamento do plano SQL continuam obrigatórios antes das runs confirmatórias. Prazo informado: **fim de novembro de 2026**. [Índice](TCC-00-LEIA-ME.md).

| Fase | Janela indicativa | Entrega e critério de saída |
|---|---|---|
| 0. Aprovação e preparação | 27/09 até início de outubro | Pedro aprovou o escopo e autorizou iniciar a implementação; Git local iniciado. Levar o plano ao orientador e registrar eventual ajuste como emenda antes das runs confirmatórias; preparar backup e publicar contratos v2 antes de modificar os contratos da POC. |
| 1. Viabilidade do instrumento | início de outubro | Em RTX 5070, provar PyTorch → ONNX → TensorRT com batch 1, E calculado no engine, L/V/S reprodutíveis e relatório de compatibilidade. Testar FP8, INT4/FP4 e 2:4 em arquiteturas representativas; opções não aprovadas ficam documentadas fora do espaço confirmatório. Sem esse gate, não iniciar busca principal. |
| 2. Núcleo científico | primeira metade de outubro | Publicar contratos v2; genoma/reparo/validação, pipelines de treino/poda/quantização, quatro métodos e comparação sob orçamento comum. Testes de propriedade, determinismo, equivalência de cache e viabilidade de engine. |
| 3. Operação e dados | outubro | Worker/fila durável, CLI, adaptação da TUI, SQLite, catálogo de artefatos e Streamlit somente de análise. Testar crash/restart, cancelamento, separação de processos, integridade e exportação. |
| 4. Piloto e congelamento | segunda metade de outubro | Rodar piloto separado; medir tempos, ruído, fidelidade de treino curto, taxa de inválidos, espaço admitido, regra de 5/8 épocas, referência de HV e viabilidade da divisão sequencial já fixada. Popular e congelar no SQLite o plano operacional com seeds, ordem, corte, tolerâncias e interrupção; exportar/hash junto de protocolo, código, splits, builder e retenção antes das oito runs. Se gate falhar, discutir emenda com orientador. |
| 5. Experimento confirmatório | fim de outubro a meados de novembro | Oito blocos pareados × quatro métodos × 5,5 h GPU ativa: ~176 h, mais overhead e reposições. Um job de GPU por vez; sessões preferidas de até 24 h, teto eventual de 48 h. Auditar sentinela, orçamento, falhas e backups a cada bloco. Não ajustar análise por desempenho observado. |
| 6. Finalistas e análise | meados de novembro | Seleção por validação de até 72 candidatos únicos (dois papéis por run, mais H1 nas oito runs conjuntas), retreino 30 épocas, congelamento de lista/artefatos, abertura única do teste oficial, cálculos de H1–H4, sensibilidade, tabelas/figuras reproduzíveis e auditoria de dados. |
| 7. Redação e revisão | em paralelo; fechar até fim de novembro | Metodologia escrita desde fase 1; resultados e discussão após fase 6; limitações, referências, anexos, reprodutibilidade, revisão com orientador e material de defesa. |

As janelas são metas de calendário, não evidência de que caibam. O piloto refaz a previsão de tempo, inclusive dos até 72 retreinos de 30 épocas. **176 h são apenas GPU ativa das buscas confirmatórias**: 7,3 dias de GPU contínua, sem piloto, retreinos, benchmark extra, interrupções, escrita e revisão. Isso deixa pouca margem até fim de novembro; se a viabilidade ou o piloto atrasarem, reduzir escopo **somente por emenda prospectiva com o orientador**, sem ocultar controles ou resultados negativos. Prioridade de corte, se exigida: opções condicionais que falhem admissão e refinamentos da interface; não sacrificar separação do teste, igualdade de orçamento, logs ou controles para “ganhar tempo”.

### Portões de qualidade entre fases

1. Instrumento: engine funcional e orçamento de ruído de L/V publicado. O mecanismo de medir V precisa passar em teste de alocação conhecida.
2. Ciência: mesmo genoma expresso produz mesmo contexto de cache; todas as quatro rotas medem o mesmo `E,L,V,S` e o custo sequencial inclui compressão.
3. Operação: retomada após queda não duplica indivíduo confirmado nem omite tempo de uma falha; TUI e CLI produzem registros equivalentes; Streamlit não escreve tabelas científicas.
4. Congelamento: hashes de código, espaço, split, protocolo e ambiente arquivados; scripts de análise executam em dados sintéticos antes de haver resultados reais.
5. Entrega: tabelas são regeneráveis a partir do export, inclusive números de falhas, dados por seed, resultados negativos e hashes dos finalistas.

Antes de cada corrida longa, fazer uma run curta integral, backup testado, verificação de temperatura/VRAM/disco, energia e retomada. A campanha longa é dividida em lotes; a persistência é por indivíduo/tentativa, não só por run. Alterar arquitetura da POC ou seus contratos requer aprovação explícita da v2.
