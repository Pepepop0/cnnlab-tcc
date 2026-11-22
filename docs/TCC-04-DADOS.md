# Dados, proveniência e auditabilidade

**Proposta 1.0 · 27/09/2026, aprovada por Pedro para implementação.** Contrato lógico do TCC; a migração ainda não foi feita. [Protocolo](TCC-01-PROTOCOLO.md) · [arquitetura](TCC-03-ARQUITETURA.md).

## Princípio de evidência

Toda afirmação comparativa precisa ser rastreável a `protocolo → ambiente → split/seed → run → candidata → tentativa → engine → medição → análise`. O dado bruto e a transformação que produziu uma tabela/figura devem existir, com hash. É proibido corrigir uma medida alterando sua linha: criar nova tentativa e registrar a regra de substituição na análise. Ausência, invalidade e falha são estados, nunca valores numéricos artificiais.

## Esquema lógico mínimo de SQLite v2

| Entidade | Chave e vínculos | Campos indispensáveis |
|---|---|---|
| `protocol_versions` | `protocol_id` | documento/hash, data de congelamento, estado `pilot|confirmed|amended`, justificativa/antecessor. |
| `experiment_plans` | `plan_id`; FK protocolo | oito blocos com seeds exatas e ordem dos métodos/runs; regra de corte das 5,5 h, excedente e retomada; tolerâncias numéricas por formato; versão/hash do plano, autor, data e estado `draft|frozen|amended`. |
| `environments` | `environment_id` | GPU, VRAM, SO, driver, CUDA, PyTorch, TensorRT, ONNX, dependências, flags do builder, relógios/política energética e hash do manifesto. |
| `datasets` | `dataset_id` | versão CIFAR-100, hashes e índices dos splits 45k/5k/10k, índices de calibração, classes, transformações. |
| `runs` | `run_id`; FK plano/protocolo/ambiente/dataset | método, seed_block, seeds distintas, space_hash, orçamento, timestamps, status, ordem no agendador. |
| `jobs` | `job_id`; FK run | estágio, prioridade/ordem, estado, heartbeat, checkpoint RNG/população, tempo ativo acumulado. |
| `candidates` | `candidate_id`; FK run | geração, pais/operador, genoma bruto/expresso canônico, reparos, phenotype_hash, cache_context_hash. |
| `evaluation_attempts` | `attempt_id`; FK candidata/job | número da tentativa, etapa, `valid|invalid|interrupted`, código de falha, duração por etapa, tempo GPU total, data; chave idempotente para confirmação. |
| `measurements` | `measurement_id`; FK tentativa | E fração, L ms, V MB, S MB, protocolo de benchmark, amostras brutas, p5/p95, precisão efetiva, indicadores de qualidade; somente se válido. |
| `artifacts` | `artifact_id`; FK tentativa/run | tipo, caminho relativo, SHA-256, bytes, engine/model fingerprint, estado presente/verificado/ausente. |
| `front_snapshots` | `snapshot_id`; FK run | tempo GPU ativo acumulado, número de avaliações únicas válidas, conjunto de `candidate_id`, HV, referência/versão. |
| `finalist_selections` | `selection_id`; FK run/candidata | papel predefinido (`menor_E`, `equilibrio`, `H1` quando aplicável), critério em validação, instante de congelamento, hash da lista, referência mono. |
| `test_evaluations` | `test_id`; FK finalista/artefato | teste top-1/top-5, matriz de confusão, por classe, seed de retreino, uma abertura por engine único. |
| `annotations` | `annotation_id`; FK opcional run/candidata | texto do usuário, autor, data, revisão. Separada das tabelas científicas. |
| `analysis_outputs` | `analysis_id` | script/hash, query, filtros, parâmetros, versão do protocolo, artefatos de tabela/figura, data. |

Tipos, constraints e migrações serão detalhados na implementação v2. O **SQLite é a fonte de verdade dos parâmetros operacionais efetivos**. `experiment_plans` pode ser editado enquanto `draft`; a versão `frozen` é imutável e precisa existir antes da primeira run confirmatória, com export e hash para auditoria fora do banco. Uma alteração cria nova versão ligada à anterior; não reescreve o plano usado por uma run. Unicidade: `run_id` identifica exatamente método × bloco × versão do protocolo/plano; `candidate_id` não é hash de arquitetura (dois sorteios iguais continuam eventos distintos); `cache_context_hash` identifica avaliação reutilizável **só dentro da run**. Um `attempt_id` não muda de estado terminal depois do commit. `measurements` só existe para tentativas válidas, com unidades explícitas. FKs ativas e schema versionado; migração nunca reinterpreta sem registrar versão e SQL aplicado.

## Evidência por avaliação

Guardar proposta genética, expressão, reparos, parâmetros/MACs, curva curta de treino, pesos/ONNX/engine quando a política de retenção pedir, inspector do TensorRT, transformações e versões, valores E/L/V/S, amostras de latência e métricas auxiliares. Para **toda candidata**, mesmo inválida: status, fase, motivo, stack trace resumida, horário e GPU-segundos consumidos. Para hit de cache: origem `attempt_id`, razão de equivalência e tempo poupado; não contar como avaliação única.

Política de retenção: manter todos os registros tabulares, logs compactos, splits, scripts, manifestos e resultados brutos de benchmark; manter `.plan`, `.onnx` e checkpoint de **todos os finalistas e candidatos das frentes/controles usados em figuras**. Para outros indivíduos, o piloto mede disco antes de decidir se guarda engine/pesos ou só hashes, logs e receita de reconstrução. Essa decisão e a possibilidade real de reconstruir devem ser documentadas antes das runs; ausência de binário antigo não pode ser escondida por um hash. Backups em unidade distinta antes e durante a campanha; testar restauração e integridade do banco.

## Transações, falhas e exportação

SQLite em WAL com um escritor e leitores de consulta; `foreign_keys=ON`, transações por indivíduo confirmado e sincronização durável configurada/testada na máquina local. Publicar arquivos grandes com escrita temporária, `fsync` quando disponível, hash e `os.replace`; depois registrar a referência em transação. Crash entre arquivo e commit deixa órfão detectável; crash depois de commit sem arquivo exige alerta de integridade. Nada de misturar caminhos absolutos em dados exportados: catálogo de artefatos usa caminhos relativos a uma raiz versionada.

Exportar em pacote com CSV/Parquet ou JSON estável, esquema, hashes e `README`: runs/attempts/measurements, amostras cruas, manifestos, scripts de análise, tabelas e figuras. Toda tabela traz `method`, `seed_block`, `protocol_stage`, versão do espaço, janela de orçamento, n válidos, n inválidos e n de cache. A POC em JSONL pode ser importada com `legacy=true` e mapeamento documentado; nunca compõe os pares H2/H3 nem as frentes do TCC, porque mudou treino, fitness e instrumento.

## Análises e gráficos pré-especificados

| Saída | Eixo/unidade e comparação | Evidência que a acompanha |
|---|---|---|
| Frentes Pareto | E × L, E × V, E × S; facetas por método/seed. | Todos os não dominados avaliados, erro do engine e versão do espaço. |
| HV ao longo da busca | HV × horas ativas de GPU (primário), HV × número de válidos únicos (secundário). | Mesma referência congelada, 8 curvas/bloco, mediana e dispersão, pontos fora de referência. |
| H1 | Diferenças em pp de acurácia, ms, MB e MB contra campeão mono da mesma seed. | Oito pares, falhas e efeitos individuais, não só média. |
| H2/H3 | Diferenças pareadas de HV no orçamento comum. | p exato/Holm, efeito e IC com método indicado, oito pares, análise de sensibilidade da referência. |
| Fidelidade e robustez | Rank 5/8 × 30 épocas; sentinela/ruído/drift por sessão; taxas de build/validade. | Amostra piloto, logs brutos, emendas e exclusões justificadas. |
| H4 | Latência real × MACs/parâmetros, resíduos por família de operação/precisão. | Split por arquitetura, não por clones, erros e intervalos. |
| Custo científico | Tentativas válidas/inválidas, cache hits, tempo treino/poda/build/bench, consumo de disco. | Mesmo orçamento e contabilidade de falhas para todos. |
| Finalistas | Trade-offs após 30 épocas e teste oficial, acurácia top-1/top-5/por classe. | Lista congelada antes do teste, todos os resultados, diferenças busca-curta × final. |

Nenhuma linha vazia é preenchida por resultado esperado. O relatório deve apresentar achados negativos e limites: único dataset/GPU/backend, proxy de treino curto, seleção de finalistas e variabilidade de compilação/temperatura. Comparações não pareadas são explicitamente rotuladas exploratórias.
