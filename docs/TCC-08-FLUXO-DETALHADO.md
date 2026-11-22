# Como o TCC funciona — fluxo detalhado de ponta a ponta

**Guia explicativo da proposta, 27/09/2026.** Este texto descreve a arquitetura-alvo; não afirma que o pipeline novo já existe. Comece pelo [mapa geral](arquitetura-tcc.html) e acompanhe o [ciclo do AG](fluxo-ag-tcc.html). As definições científicas vinculantes estão no [protocolo](TCC-01-PROTOCOLO.md); os domínios exatos, no [genoma](TCC-02-GENOMA.md).

## 1. O que estamos tentando descobrir

Não queremos apenas uma CNN com alta acurácia. Queremos investigar se **escolher arquitetura e representação ao mesmo tempo** permite encontrar soluções com boa classificação do CIFAR-100 e menor custo **quando a rede já está pronta para inferência**. Custo tem três aspectos distintos: latência (`L`), pico incremental de memória (`V`) e tamanho em disco do engine (`S`). A qualidade é o erro top-1 (`E`). O resultado da pesquisa não precisa ser uma única “melhor rede”: é uma **frente de compromisso**, na qual melhorar uma propriedade normalmente exige ceder em outra.

O elemento medido é sempre `M_g`, o engine TensorRT construído a partir da candidata `g`. Não basta contar parâmetros ou MACs, nem medir o checkpoint PyTorch antes da poda/quantização. Esses valores são diagnósticos úteis, mas não substituem a fitness real. O estudo é delimitado a CIFAR-100, uma RTX 5070, batch 1 e ambiente/versões registrados; não demonstra ganho universal em qualquer GPU ou tarefa.

## 2. Onde exatamente entra o AG

O **algoritmo genético (AG)** está no **core da busca**, entre o worker que controla a execução e o pipeline que avalia as candidatas. Ele não é a TUI e não é o otimizador de pesos da CNN.

- O **worker** reserva a única GPU, conduz a fila de jobs, retoma trabalho interrompido e conta as 5,5 h de GPU ativa da run. Não decide qual rede vence.
- O **AG** mantém população e arquivo Pareto, propõe genomas, recebe uma fitness de quatro números para cada rede válida, seleciona pais, aplica crossover/mutação e repete. No método principal, sua regra de seleção é NSGA-II.
- O **treino por gradiente** acontece *dentro da avaliação de cada genoma*: AdamW ajusta os pesos daquela CNN por 5 ou 8 épocas de busca. O AG altera a **receita/topologia/representação entre redes**; o gradiente altera os **pesos dentro de uma rede**. Há, portanto, duas escalas de otimização aninhadas.
- O **avaliador** transforma pesos treinados em engine real e devolve `E,L,V,S`. Ele é compartilhado entre os quatro métodos para evitar que cada controle use uma régua diferente.

O ciclo essencial é: `AG propõe g → g vira CNN → CNN é treinada → podada/quantizada/exportada/compilada → engine é medido → F(g) volta ao AG → AG seleciona e propõe a próxima geração`. Essa seta de retorno da fitness para a seleção é a parte que estava pouco visível no diagrama geral.

## 3. Preparação antes da primeira candidata

1. **Congelar o protocolo e o espaço:** versão/hash dos documentos, código, builder TensorRT, genoma admitido e regras dos quatro métodos. O piloto, separado das oito seeds confirmatórias, pode decidir opções condicionais e 5 versus 8 épocas **antes** da campanha principal.
2. **Congelar os dados:** 45.000 imagens oficiais de treino para pesos, 5.000 de validação para fitness/seleção e 10.000 do teste oficial lacradas até os finalistas. Os 1.024 exemplos de calibração PTQ são retirados apenas dos 45.000 de treino. Persistir os índices e hashes.
3. **Preparar oito blocos de seeds:** cada bloco reúne os quatro métodos sob mesma seed de treino/split/ambiente, com randomização de busca registrada. A réplica estatística é a **run por método e bloco**, não cada indivíduo.
4. **Preparar instrumento e operação:** verificar pipeline PyTorch→ONNX→TensorRT na máquina, ruído da latência, pico V, temperatura, disco, backup e recuperação. Se o instrumento não medir uma das quatro propriedades com confiança, não iniciar as runs confirmatórias.

## 4. Da TUI até a população inicial

Na TUI, Pedro escolhe um job predefinido pelo protocolo (piloto, run confirmatória ou finalista) e vê fila/progresso. A CLI headless deve poder enviar o mesmo tipo de job. A fila durável entrega ao worker quando a GPU estiver livre; não há dois treinos/benchmarks concorrendo na mesma placa. O worker cria a run no banco com método, bloco, seeds, hashes, orçamento, ambiente e estado. Eventos de progresso retornam à TUI sem que o core acesse widgets.

Para o método conjunto, o NSGA-II cria uma população inicial de **30 genomas** por amostragem válida. Cada genoma é uma receita, não um conjunto de pesos prontos: contém número e características dos blocos, opções de treino e modos de representação/poda. O genoma de comprimento fixo tem seis posições de bloco, das quais 2 a 6 estão ativas. Os genes latentes são guardados para operadores evolutivos, mas não alteram o fenótipo enquanto inativos.

**Exemplo ilustrativo, não resultado medido:** um indivíduo pode pedir quatro blocos, um bloco depthwise-separável, largura de 64 canais, precisão FP16 em partes da rede e poda física de 25% em outro bloco. Isso ainda não diz sua acurácia nem seu ganho. Só depois de construir e executar o engine saberemos se o pedido se expressou de fato e se foi vantajoso.

## 5. O que acontece com *um* indivíduo

| Etapa | Trabalho e pergunta respondida | Evidência registrada |
|---|---|
| 1. Proposta | AG amostra ou produz por crossover/mutação um `raw_genome`; registrar geração, pais e operador. | Genoma bruto, RNG/seed, instante e método. |
| 2. Reparo | Aplicar regras determinísticas: limite de redução espacial, canais compatíveis, residual/projeção, genes condicionais. Um genoma impossível fica inválido com motivo. | `expressed_genome`, reparos e classe de invalidez. |
| 3. Cache | Verificar se a mesma expressão **e todo o contexto** já foram avaliados nesta run. Hit não vira nova avaliação nem ponto independente. | `cache_context_hash`, origem da medição ou miss. |
| 4. Construção/treino | Instanciar CNN, aplicar split/augmentação fixos, AdamW e treino curto (5 ou 8 épocas decidido pelo piloto). | Curva, tempo, seed e checkpoint conforme retenção. |
| 5. Poda e ajuste | Se o genoma solicitar, remover canais fisicamente e corrigir dependências; uma época de ajuste. Se não preservar a topologia, invalidar. | Canais pedidos/reais, grafo dependente, falha ou pesos ajustados. |
| 6. Quantização | PTQ calibra com treino; QAT acrescenta época de ajuste quando pedido. Gerar Q/DQ explícito e respeitar compatibilidade por camada. | Política, amostras de calibração, precisões solicitadas/reais. |
| 7. Exportação | Exportar ONNX, verificar tensores e equivalência de amostra. | Arquivo/hash, operador problemático e diferença numérica. |
| 8. Compilação | Construir `.plan` TensorRT com builder fixo; inspecionar tipos/táticas e rejeitar promessa não materializada. | Engine/hash, inspector, flags e tempo de build. |
| 9. Validação do engine | Inferir as 5.000 imagens de validação **no engine**, não no PyTorch, e obter erro top-1. | `E`, top-5 auxiliar, divergências e classes. |
| 10. Benchmark isolado | Desalocar treino e medir batch 1 em processo novo: L por eventos GPU, V como pico incremental, S pelos bytes do `.plan`. | Amostras brutas, mediana/p95, baseline de memória, bytes, sentinela e temperatura. |
| 11. Commit | Uma transação associa candidata, tentativa, medição e artefatos. Se falhar em qualquer etapa, persistir motivo e GPU-segundos; não fabricar fitness. | `attempt_id`, status, tempos, hashes e `E,L,V,S` somente se válido. |

O loop de avaliação é caro porque cada candidato pode exigir treino, poda, QAT, exportação, build e benchmark. Por isso a comparação primária entre métodos usa **tempo ativo de GPU**, não somente quantidade de avaliações. Falhas e tentativas inválidas consomem o orçamento e são resultados operacionais relevantes.

## 6. Como os quatro números voltam para o AG

Um candidato válido retorna `F(g)=(E,L,V,S)`, com tudo a minimizar. `E` é fração de erros top-1 nas 5.000 imagens de validação do engine; `L` é mediana de latência GPU batch 1 em ms; `V` é MB decimais de pico incremental; `S` é MB decimais do `.plan`. As unidades, aquecimento, repetições e auditorias estão no protocolo. Um indivíduo inviável não ganha `E=1` ou algum custo fictício: tem estado **inválido** e diagnóstico. A seleção prefere viáveis; a taxa de invalidade é apresentada separadamente.

No NSGA-II, um indivíduo **domina** outro quando não é pior em nenhum dos quatro objetivos e é melhor em pelo menos um. Os não dominados formam a primeira frente; camadas seguintes têm prioridade menor. Quando uma frente não cabe inteira na população, a regra de diversidade (*crowding distance*) evita concentrar todos os pais numa única região. Isso permite que uma CNN um pouco menos precisa, mas muito menor/mais rápida, continue interessante sem escolher pesos arbitrários para somar as quatro métricas.

Após avaliar indivíduos, o AG mantém o arquivo externo com **todos os viáveis não dominados já avaliados**, não só os 30 indivíduos atuais. Seleção gera pais; crossover troca genes/blocos e mutação altera genes categóricos, ordinais ou contínuos dentro dos domínios. O reparo torna a nova proposta válida quando possível; cache evita retreino da mesma expressão/contexto. A nova geração volta à etapa 1. O processo prossegue até o orçamento da run. **Número de gerações e de indivíduos efetivamente avaliados são resultados observados**, não quotas iguais pressupostas.

## 7. Onde entram os controles experimentais

O objetivo do TCC não é somente mostrar que “o AG encontra redes boas”; precisamos saber *por que* e *em relação a quê*. Todos os controles usam split, instrumento, espaço aplicável e limite de 5,5 h por seed.

| Método | O que muda na escolha de candidatos | O que continua igual |
|---|---|---|
| **Conjunto (principal)** | AG NSGA-II seleciona com quatro objetivos e evolui arquitetura + representação ao mesmo tempo. | Avaliador, orçamento e registros. |
| **Monoobjetivo** | AG no mesmo espaço evolui selecionando só pelo menor erro `E`; mede também L/V/S, sem usá-los na seleção. | O mesmo engine e as mesmas quatro medidas. |
| **Aleatório** | Não há AG: amostragem independente do mesmo espaço válido, com deduplicação. | Avaliação idêntica; mostra o valor adicional da busca orientada. |
| **Sequencial** | AG NSGA-II busca arquitetura em FP32 por 2,75 h; congela até sete arquiteturas; outro estágio evolui representação/poda nelas por 2,75 h. | 5,5 h totais, mesma instrumentação e opções de representação admitidas. |

Os quatro métodos × oito blocos dão **32 runs confirmatórias** e estimativa de **176 h de GPU ativa**, fora piloto/finalistas. O agendador alterna métodos por lotes para reduzir viés de ordem/temperatura. A comparação principal H2/H3 é o *hypervolume* (HV) do arquivo Pareto ao mesmo corte de horas GPU, com ponto de referência congelado após o piloto; também relatamos HV por número de candidatos válidos únicos. Aleatório e mono são controles importantes mesmo que não forneçam um “melhor” em todas as métricas. A POC anterior é referência histórica, não uma quinta run pareada.

## 8. Encerramento da busca e teste oficial

Ao fim das oito seeds, o arquivo de cada run contém muitos pontos de busca **curta**. Para não transformar o teste oficial em instrumento de escolha, a seleção de finalistas é feita **somente pela validação** e tem papéis fixos: menor `E` e ponto de equilíbrio por run; nas oito runs conjuntas, mais um candidato H1 dentro de 1 pp de `E` do campeão mono da mesma seed, se existir. Duplicatas entre papéis são retreinadas uma vez. O limite teórico é **72 finalistas únicos**.

Cada finalista é recomeçado do zero, treinado por **30 épocas** com a política final igual entre métodos, passa de novo pelas transformações, compilação e medições L/V/S. A lista/artefatos ficam congelados e registrados antes de abrir as 10.000 imagens do teste oficial. Cada engine final único recebe **uma** avaliação do teste; se um resultado cair, não se escolhe outro finalista depois. H1 compara o candidato conjunto pré-escolhido com o campeão mono da **mesma seed**, em acurácia top-1 de teste e nos três custos. H2/H3 são análises pareadas das frentes de busca no orçamento comum. H4, sobre MACs/parâmetros versus latência, é exploratória.

## 9. O que fica salvo e como se apresenta

O SQLite v2 deve guardar protocolo, ambiente, split, seed, método, job, genoma bruto/expresso, tentativas, medições, snapshots da frente, seleção de finalistas e resultados de teste. Checkpoints, ONNX, `.plan`, amostras brutas e figuras ficam em arquivos com hash/caminho no catálogo. Persistimos **por indivíduo/tentativa** para retomar após queda e para auditar falhas; uma remedição cria outra tentativa, nunca apaga a primeira.

A **TUI** cria e acompanha treinos; o **Streamlit** lê o banco para comparar frentes, curvas HV, custo por etapa, taxa de falhas e finalistas, exportando figuras/tabelas. Pode guardar anotações em tabela separada; não altera `E,L,V,S`, genomas, seeds ou resultados. O [plano de dados](TCC-04-DADOS.md) e o [modelo de resultados](TCC-07-RESULTADOS.md) especificam as saídas que deverão constar do TCC, inclusive resultados negativos e limites.

## 10. Forma curta de explicar ao orientador

> “O AG é o controlador da busca: ele cria receitas de CNN, recebe de volta quatro medidas do **engine realmente executado** e evolui uma população de compromissos entre erro, latência, memória e armazenamento. Cada candidato passa por treino, poda/quantização, exportação, TensorRT e benchmark. Comparamos essa busca conjunta a evolução focada só em erro, amostragem aleatória e otimização sequencial, sempre com o mesmo tempo de GPU por seed. Só depois escolhemos finalistas por validação, retreinamos e abrimos o teste oficial. Toda decisão, falha e medida fica rastreável.”

## 11. O que ainda não é fato experimental

TensorRT funcionar bem para cada operação, FP8/4-bit/2:4 trazer ganhos, cinco épocas ranquearem como trinta, V ser estável e os 72 retreinos caberem até novembro são **hipóteses de viabilidade**, não resultados. O piloto mede isso. Se uma condição falhar, a regra é documentar e discutir emenda com o orientador **antes** das runs confirmatórias; não mudar o espaço, a fitness ou os controles depois de conhecer quem venceu.
