# Genoma, transformação e avaliação de candidatas

**Versão:** proposta 1.0, 27/09/2026. Vinculada ao [protocolo](TCC-01-PROTOCOLO.md). Todos os domínios e operadores são iguais nos métodos que usam o genoma completo; o método sequencial acessa os mesmos domínios de representação após escolher arquiteturas.

## 1. Princípio

O genoma descreve a arquitetura `A_g`, a política de treino e a representação final. A unidade avaliada é `M_g`, o engine TensorRT após treino, remoção física de canais, quantização e compilação. Genes que só mudam uma propriedade no papel, sem alterar um engine executável e ao menos uma medida L/V/S acima do ruído, **não são admitidos**. O erro E é medido sempre depois das transformações.

O vetor tem comprimento fixo com seis blocos disponíveis. `n_blocks` ativa os primeiros 2 a 6; os demais ficam latentes para mutação/crossover, porém não entram no hash do fenótipo enquanto inativos. O genoma bruto também é guardado para reconstruir a evolução.

## 2. Domínios fechados para o piloto

| Grupo | Genes | Domínio |
|---|---|---|
| Global — arquitetura | `n_blocks`; canais do stem; ativação; cabeça | `{2,3,4,5,6}`; `{16,32,64}`; `{ReLU, SiLU, Hardswish}`; `{GAP+linear, GAP+FC256+linear}` |
| Global — representação | precisão do stem e da cabeça; modo INT8; granularidade dos pesos; calibração PTQ | `{FP32,FP16,BF16,INT8}` cada; `{PTQ,QAT}`; `{por tensor,por canal}`; `{max,entropia,p99,99}` |
| Global — treino | learning rate; weight decay; dropout da cabeça | `log10(lr)∈[-4,-2]`; `log10(wd)∈[-6,-3]`; `0 a 0,5` |
| Por bloco — arquitetura | canais; operação; kernel; convoluções; residual; redução espacial | `{16,32,48,64,96,128,192,256}`; `{padrão,agrupada g=4,depthwise-separável,MBConv e=3,MBConv e=6}`; `{3,5,7}`; `{1,2,3}`; `{não,sim}`; `{nenhuma,maxpool,avgpool,conv stride 2}` |
| Por bloco — representação | precisão; poda estruturada | `{FP32,FP16,BF16,INT8}`; `{0%,25%,50%}` de canais, com remoção física |
| Condicionais | FP8; INT4/FP4 em pesos; 2:4 | Somente após teste de admissão abaixo; acrescentados à versão congelada do espaço **antes** das runs principais. |

Os valores condicionais não são prometidos como “convolução INT4”: TensorRT descreve INT4 como representação somente de pesos, com desquantização para execução. FP8 e formatos de 4 bits dependem da operação e do esquema Q/DQ, além do suporte geral ao tipo numérico. [Tipos](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html), [esquemas](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).

Itens fixos: BatchNorm com fusão verificada no engine, AdamW, batch de treino 128, cosine com uma época de aquecimento, flip horizontal e crop 32×32 com padding 4, AMP BF16 durante treino se o teste local confirmar equivalência, entrada de inferência FP32 e batch 1. A POC implementa apenas flip; crop pertence à nova implementação. Outras normalizações, poda não estruturada, clustering/codebooks, truncamento livre de mantissa e low-rank pós-treino ficam fora do espaço pelos motivos da proposta de 23/09.

## 3. Regras de construção e reparo

1. No máximo quatro reduções espaciais ativas; manter 32→16→8→4→2 pixels. Excedentes são convertidas em `nenhuma` a partir do último bloco, em ordem determinística.
2. Canais de entrada/saída para agrupada `g=4` permanecem divisíveis por quatro; projeção residual 1×1 é criada quando largura ou resolução muda. Ela é parte do modelo e do custo medido.
3. Poda física por bloco remove canais e atualiza camadas dependentes, inclusive caminhos residuais e grupos. O número efetivo é arredondado para múltiplo de 16; registrar proporção solicitada e realizada. Quando não for possível preservar a topologia, marcar inválido com causa. A [Dependency Graph](https://github.com/VainF/Torch-Pruning/blob/master/README.md) é a referência de implementação, sujeita ao teste local.
4. Genes de quantização não expressos ficam fora do hash. Quantização INT8 é simétrica, com Q/DQ explícito e calibração só no treino. Precisão por bloco precisa ser materializada no grafo, com casts nas fronteiras; o engine inspector confirma o tipo realmente executado. [TensorRT: precisão](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html).
5. Esparsidade 2:4 só é admissível em convolução/matriz com dimensão de redução divisível por 4, pesos no padrão 2:4, precisão compatível e camada selecionada para tática esparsa no engine. **Depthwise comum não atende a condição de quatro canais por grupo**; é inválida para esse gene. Registrar separadamente “elegível”, “tática escolhida” e ganho medido. [TensorRT: requisitos 2:4](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html).
6. Reparo é puro e determinístico, anterior ao treino. O registro contém `raw_genome`, `expressed_genome`, lista de reparos e razão de invalidez. Uma candidata inviável não recebe valores fictícios de E/L/V/S.

## 4. Operadores e cache

Amostragem inicial uniforme por gene **condicionalmente válido**; contínuos uniformes na escala log definida. Crossover por bloco inteiro e por gene global; mutação ordinal de um passo, categórica por reamostragem e contínua em log. Probabilidades iniciais: crossover 0,6, mutação 0,3, sigma relativo 0,15; população 30. Elites carregam genoma **e fitness medida**, sem novo treino. Candidatos repetidos têm até 10 tentativas de reamostragem; depois a tentativa duplicada fica registrada e a busca prossegue sem contabilizá-la como nova avaliação.

O cache usa genoma expresso canônico, seed de treino, fingerprint do split, política de treino, poda, quantização, código, backend e builder config. Um hit dentro da mesma run não custa GPU nem aumenta o contador de avaliações únicas. **Não compartilhar fitness entre seeds** apenas porque a arquitetura é igual. Toda falha de build/treino registra tempo gasto, classe do erro e fase, para analisar taxa de validade do espaço.

## 5. Pipeline de avaliação

`amostrar → reparar → verificar cache → construir → treinar → podar/ajustar → quantizar/ajustar → exportar ONNX → compilar TensorRT → verificar engine → medir E,L,V,S → persistir`.

Treino curto: 5 épocas, ou 8 se o piloto indicar baixa fidelidade; protocolo idêntico para os quatro métodos. Se houver poda, uma época de ajuste fino; se houver QAT, uma época de ajuste fino com quantização simulada após a poda. PTQ usa os 1.024 exemplos congelados do conjunto de treino. Esses custos adicionais entram no orçamento de GPU. O teste de acurácia do engine se faz em 5.000 imagens de validação durante a busca. O teste oficial de 10.000 só roda após os finalistas serem congelados e retreinados por 30 épocas.

Verificações antes de aceitar uma avaliação: formato e valores finitos; inferência PyTorch/ONNX/engine com tensores de amostra; tipo efetivo de precisão no engine; diferença numérica da exportação dentro de tolerância publicada para cada formato; ausência de dados de teste no grafo de treino/calibração. Divergência que altera a classe prevista é incorporada a E; falha estrutural ou ausência da precisão solicitada torna o genoma inválido.

## 6. Teste de admissão das técnicas condicionais

Antes das runs confirmatórias, congelar três arquiteturas representativas do piloto (pequena, média, grande) e medir cada transformação contra a mesma arquitetura/pesos sem ela. Para entrar no espaço, a técnica precisa: **(a)** construir e executar nos casos de operação para os quais será declarada válida; **(b)** comprovar no engine o formato/tática proposto; e **(c)** melhorar L ou V além de duas vezes a variação máxima observada entre dez remedições da sentinela, ou diminuir S por bytes exatos, em pelo menos duas das três arquiteturas representativas sem falha. Reportar qualquer perda de E. Se a técnica só servir a uma operação, o gene fica condicionado à operação, não habilitado globalmente.

Resultados negativos também são tabelados. O espaço admitido recebe identificador e hash antes das oito seeds. Uma mudança posterior exige emenda e, se afetar comparabilidade, repetição das runs. Isso preserva o escopo completo da proposta sem afirmar que toda opção enumerada pelo TensorRT seja vantajosa numa CNN 32×32.
