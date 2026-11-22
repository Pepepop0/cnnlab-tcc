# Protocolo de pesquisa — CNNs eficientes para inferência

**Versão:** proposta 1.0, 27/09/2026. **Status:** aprovada por Pedro para implementação; aceite formal do orientador não registrado e resultados empíricos pendentes. Ver [índice](TCC-00-LEIA-ME.md) e [registro de decisões](TCC-06-DECISOES.md).

## 1. Pergunta e limites

**Pergunta:** uma busca evolutiva multiobjetivo que escolhe conjuntamente arquitetura e representação da rede encontra, em uma RTX 5070, CNNs para CIFAR-100 que conservam a qualidade de classificação com menor latência, pico de memória e armazenamento do artefato executável?

O objeto científico é a **rede pronta para inferência**. Treinamento, energia, custo financeiro e throughput não integram a fitness. Os resultados se aplicam ao espaço de busca, split, máquina, versões e protocolo registrados; não implicam vantagem universal em outras GPUs ou datasets. Bayesian Optimization saiu do escopo após o pivô. Comparar métodos de busca de hiperparâmetros já não é a pergunta central.

### Hipóteses e interpretações pré-definidas

| ID | Enunciado | Avaliação |
|---|---|---|
| H1 | A fronteira conjunta oferece redução simultânea de L, V e S sem perder mais de 1 ponto percentual de acurácia. | Para cada seed, comparar finalistas pré-selecionados por validação ao campeão monoobjetivo da mesma seed, ambos retreinados. Critério descritivo de apoio: sucesso em pelo menos 6 das 8 seeds. Reportar também as oito diferenças e os casos de falha. |
| H2 | A busca conjunta encontra frentes melhores que amostragem aleatória. | Diferença pareada de hypervolume no orçamento comum de tempo; teste unilateral exato, Holm com H3, alfa 0,05. |
| H3 | Busca conjunta supera a sequência “buscar arquitetura, depois comprimir”. | Mesma análise de H2, contando a compressão posterior no tempo do método sequencial. |
| H4 | MACs e parâmetros, isoladamente, explicam apenas parte da latência. | Análise **exploratória**: correlação e modelo preditivo testado em arquiteturas não usadas no ajuste. Sem limiar rígido de R² como critério de aprovação do TCC. |

H1 é deliberadamente exigente; ausência de solução é conclusão válida. H2 e H3 devem mostrar tamanho do efeito e resultados individuais além do p-valor. A seleção de finalistas nunca consulta o teste oficial.

## 2. Dados e unidade experimental

- CIFAR-100: 50.000 imagens oficiais de treino e 10.000 oficiais de teste. Reutilizar a separação estratificada determinística da POC: 45.000 para ajustar pesos e 5.000 para validar indivíduos, com todas as 100 classes. Congelar e registrar índices, classes e hashes antes da busca.
- O teste oficial de 10.000 imagens fica lacrado até a lista de finalistas estar gravada. Calibração PTQ usa 1.024 exemplos **apenas dos 45.000 de treino**, com índices fixos e cobertura das classes.
- A unidade de repetição estatística é uma **run por método e seed**; milhares de indivíduos dentro da mesma run não são milhares de réplicas independentes.
- Oito seeds da busca, pareadas entre os quatro métodos por bloco. Seed de inicialização/treino fixa dentro de cada bloco para reduzir ruído; diferente entre blocos. Registrar separadamente seeds de busca, treino, embaralhamento, calibração e retreino. Os **valores exatos e a ordem** serão gravados em um plano versionado no SQLite antes da primeira run confirmatória. Não reutilizar o resultado de uma seed como se fosse outra.
- Uma candidata é identificada pelo genoma **expresso após reparo** e pelo contexto: split, seed de treino, código, hiperparâmetros, transformações e backend. Cache só vale para contexto idêntico; hits não contam como novas avaliações.

## 3. Fitness e viabilidade

Seja `g` um genoma válido e `M_g` o engine que será de fato executado após treino, poda/ajuste, quantização, exportação e compilação. A fitness é a minimização vetorial:

\[
\min_{g\in\mathcal{G}}\;\mathbf{F}(g)=\big(E(g),L(g),V(g),S(g)\big).
\]

| Objetivo | Definição operacional | Unidade |
|---|---|---|
| `E` | Fração de erros top-1 nas 5.000 imagens de validação, inferidas pelo próprio engine com batch 1. | fração |
| `L` | Mediana do tempo **GPU** de uma inferência batch 1, no engine, após aquecimento; entrada FP32 1×3×32×32 já residente na GPU. Conversões internas de tipo entram. | ms |
| `V` | Pico incremental de memória da execução em processo isolado: engine carregado, contexto, buffers de I/O, recursos do CUDA Graph, ativações e workspace, acima do baseline do processo com contexto CUDA inicializado. | MB decimais |
| `S` | Bytes exatos do `.plan` serializado que executa `M_g`, divididos por 10⁶. | MB decimais |

O benchmark também registra latência ponta a ponta com transferências e pré/pós-processamento, p5/p95/IQR, memória discriminada, bytes ONNX, parâmetros, MACs, duração do treino/build, acurácia top-5 e métricas por classe dos finalistas. São auxiliares: não se somam à fitness. Nenhum checkpoint FP32 intermediário recebe crédito por compressão que não apareça em `M_g`.

Genomas cuja rede não treina, não exporta, não compila, excede a VRAM ou produz valores não finitos ficam com `status=invalid`, motivo e tempo gasto. Indivíduos viáveis sempre prevalecem na seleção. Falhas **não desaparecem** do denominador de tentativas ou do orçamento.

### Protocolo de benchmark

Benchmark serial, em processo separado do treino, sem dataset inteiro residente na VRAM. Mesma RTX 5070, driver, versão TensorRT, builder config, modo FP32/TF32, entrada, batch, contexto, stream e política CUDA Graph para todos. Inicialmente 200 inferências de aquecimento e 1.000 medidas por engine; o piloto pode elevar as medidas para 3.000 se dez sessões da rede sentinela produzirem coeficiente de variação de medianas acima de 2%. Se continuar instável, suspender as runs definitivas e versionar nova medição.

Medir `L` por CUDA events sem sincronização por inferência; comparar por mediana de sessões e registrar amostras brutas e relógios/temperatura. Uma rede sentinela é remedida a cada 50 avaliações; drift superior a 5% abre nova tentativa de medição ligada ao original. Nunca substituir a medida antiga silenciosamente. Para `V`, instrumentar alocações do TensorRT e da aplicação e conferir o pico contra a variação de memória livre; um desvio material não explicado bloqueia a utilização de `V` como fitness. `S` vem do arquivo pronto, não da soma teórica dos pesos. Os `.plan` são específicos ao ambiente, que integra o manifesto.

## 4. Quatro métodos e orçamento comparável

| Método | Espaço e seleção |
|---|---|
| AE principal | NSGA-II, quatro objetivos, população 30; arquivo externo com todos os viáveis não dominados já avaliados. |
| Monoobjetivo | Mesmo espaço, genoma, reparo e avaliação; seleciona só pela menor `E`. Mede L, V e S de cada indivíduo para comparação. |
| Aleatória | Amostragem independente do **mesmo espaço válido**, com reparo, deduplicação e avaliação idênticos. Sem seleção evolutiva. |
| Sequencial | NSGA-II de arquitetura com representação FP32 fixa por 2,75 h; congela até sete arquiteturas distintas de menor E pela validação e, nas 2,75 h seguintes, usa NSGA-II nos genes de representação e poda dessas arquiteturas, com alocação rotativa. A segunda etapa acessa o **mesmo conjunto admitido** do método conjunto. **Busca e compressão posterior gastam o mesmo orçamento total de 5,5 h.** |

A CNN manual da POC é referência histórica: reconstruída e avaliada pelo protocolo novo; não substitui nenhum dos quatro métodos nem entra nos testes H2/H3. As runs antigas com treino diferente não são misturadas às amostras confirmatórias.

**Orçamento primário:** até **5,5 horas de GPU ativa por método/seed**, incluindo treino, poda, QAT, exportação, build, medição e tentativas inválidas. Espera-se aproximadamente 22 horas de GPU para quatro métodos por bloco e **176 horas para oito blocos**, mais piloto e avaliação final. É previsão, não medição. O sequencial recebe 2,75 h para arquitetura e 2,75 h para compressão; tempo não usado numa etapa não é convertido em avaliações gratuitas. Reportar, em paralelo, HV por **tempo ativo de GPU** e por **número de indivíduos válidos únicos**.

O agendador executa um job de GPU por vez, alterna métodos entre lotes/gerações e roda as oito seeds em ordem predefinida. Espera-se terminar cada bloco em até 24 h de relógio; se medição/persistência alongar o bloco, dividi-lo sem mudar o orçamento científico. Uma sessão não supervisionada pode durar até 48 h quando necessário. O tempo ocioso entre sessões e a indisponibilidade da máquina não consomem orçamento; o trabalho já gasto por uma tentativa que falhou consome. A regra operacional de corte para uma avaliação em andamento, de contabilização de eventual excedente e de retomada após interrupção será gravada e congelada no SQLite antes das runs confirmatórias. Para curvas de HV, registrar o último valor confirmado em cada instante; se uma run terminar antes de 5,5 h, o último valor é mantido até o corte, com tempo não usado declarado. É proibido mudar orçamento após ver o desempenho de qualquer método.

## 5. Pilotagem fechada antes das runs confirmatórias

O piloto, isolado dos oito blocos confirmatórios, usa amostra fixa de até 64 genomas/combinações representativas e 20 genomas treinados por 5 e 30 épocas. Entrega tempos reais por etapa, validade de exportação, ruído L/V, correlação de rank da acurácia curta com a longa e mapa de admissibilidade. Os candidatos do piloto **não são transferidos para as frentes confirmatórias**.

Regras: `rho_Spearman(5,30) ≥ 0,7` conserva 5 épocas de treino na busca; abaixo disso, a busca inteira usa 8 épocas. Se 8 épocas também produzir rank instável no conjunto piloto, interromper e emendar o protocolo com o orientador. `V` permanece objetivo se passar na auditoria de medição; correlação `V×S > 0,95` é relatada junto com a mudança de dominância, **sem exclusão automática**. O NSGA-II permanece método principal; saturação da primeira frente vira resultado/limitação, sem troca retrospectiva para NSGA-III.

O ponto de referência do hypervolume é fixado **após o piloto e antes dos oito blocos**: `r_E=1`, `r_j=1,25 × max_j` dos genomas viáveis do piloto para `j∈{L,V,S}`. Se candidato posterior ultrapassar `r` em qualquer coordenada, não contribui para HV; contar e listar esses candidatos. Recalcular o resultado com `r'_j=1,5 × r_j` como sensibilidade, mantendo ambos os pontos iguais para todos os métodos. Não ajustar `r` para favorecer uma run. HV é calculado sobre o arquivo de todos os avaliados não dominados, não apenas a última população.

O piloto também determina a tabela de admissão do genoma, a configuração única do builder e a projeção revisada de horas. A divisão 2,75 h + 2,75 h do sequencial já está fixada; eventual inviabilidade exige **emenda anterior às runs confirmatórias**. O protocolo e código recebem identificador imutável.

## 6. Finalistas, teste e inferência estatística

De cada run, escolher pela validação **dois papéis** do arquivo de avaliados viáveis não dominados: (i) menor E; (ii) ponto de equilíbrio, menor distância euclidiana à origem após normalizar cada objetivo por `(f_j - m_j)/(r_j - m_j)`, onde `m_j` é o mínimo viável do piloto e `r_j` o ponto de referência de HV congelado. Desempatar por menor E e depois pelo hash do genoma expresso. Para a run conjunta, selecionar ainda **um candidato H1**: dentre os pontos com `E ≤ E_mono + 0,01` na validação curta da mesma seed, o menor valor de `L/r_L + V/r_V + S/r_S`; desempatar por menor E e hash. Sem ponto elegível, marcar H1 como falha naquela seed. Limiares de 2 e 5 pp são análises de sensibilidade, não uma forma de escolher resultados após ver o teste. Genomas duplicados entre papéis são retreinados uma só vez. São **no máximo 72 retreinos** (2×4×8 + 1×8), mais repetições estritamente justificadas por falha técnica e registradas; o piloto estima se cabem no calendário.

Finalistas e campeões mono são treinados **do zero**, com 30 épocas, mesmo split e política de treino final, e passam novamente por poda, QAT/PTQ, ONNX, build e benchmark. Fixar seed de retreino por bloco e compartilhar entre métodos; registrar seu valor. Congelar a lista e os artefatos antes de abrir o teste oficial. Avaliar o teste **uma vez por artefato único**, publicar todos os resultados, inclusive quedas de acurácia. Resultados de busca curta e resultados finais têm `protocol_stage` diferentes e não são misturados numa mesma frente sem rótulo.

H1 usa o campeão mono da **mesma seed** como referência, com Δacurácia **top-1 no teste oficial** em pontos percentuais; comparar L, V e S dos engines finais medidos pelo mesmo protocolo. A escolha do candidato H1 ocorre antes do teste, por validação curta, e não é trocada se o retreino o prejudicar. Para H2/H3, calcular diferença pareada de HV por seed no orçamento comum; usar teste de permutação pareado exato, unilateral (`NSGA-II completo > controle`) com correção de Holm para dois testes. Reportar oito pares, diferença média e mediana, amplitude e intervalo de confiança com método declarado. Se houver empates, o teste de permutação mantém regra exata documentada. H4 usa arquitetura/genoma efetivo como grupo na avaliação preditiva, para não tratar clones como amostras independentes.

## 7. Reprodutibilidade e referência metodológica

Cada run persiste protocolo, código, dependências, equipamento, split, seeds, espaço de busca, operador, tempos de GPU, genomas propostos/expressos, cache, falhas, medições e caminho/hash dos artefatos. O [modelo de dados](TCC-04-DADOS.md) especifica as relações. Esta versão do diretório não possui um repositório Git detectável; providenciar controle de versão antes das runs para cumprir a exigência de revisão exata do código.

Fontes: [TensorRT benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html), [precisão e build](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html), [SQLite WAL](https://www.sqlite.org/wal.html), [busca aleatória em NAS](https://proceedings.mlr.press/v115/li20c.html), [NSGA-II e crowding](https://pymoo.org/algorithms/moo/nsga2.html), [hypervolume e ponto de referência](https://pymoo.org/misc/indicators.html), [teste de permutação pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [ShuffleNet V2 e latência real](https://arxiv.org/abs/1807.11164).
