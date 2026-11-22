# Auditoria de V, 06/10/2026

**Pergunta.** O protocolo (TCC-01, seção 3) exige conferir o pico de memória medido pelo
alocador instrumentado contra a variação de memória livre; um desvio material não
explicado bloquearia V como objetivo. Na sentinela, o alocador deu 2,53 MB e a memória
livre caiu 33,55 MB. Nas 327 avaliações da fila, a diferença ia de 11 a 33 MB e crescia com
o tamanho da rede (Spearman 0,45 com V).

**Como foi medido.** `tools/piloto/auditoria_v.py`: para cada engine, um processo novo lê a
memória livre depois de cada etapa (contexto CUDA, runtime do TensorRT, deserialização,
contexto de execução, buffers e 200 execuções), com o alocador padrão do TensorRT, sem
passar pelo do PyTorch. 24 engines guardados da fila, espalhados por toda a faixa de V, e a
sentinela. RTX 5070, TensorRT 11.3, driver 610.88.

**Resultado.**

1. Abrir **qualquer** engine reserva 33,55 MB e a primeira execução reserva 12,58 MB,
   idênticos nos 25 engines, do menor (0,26 MB de arquivo) ao maior (14,78 MB). É reserva
   fixa do TensorRT, igual para toda candidata: não muda nenhuma comparação de dominância
   nem de hipervolume.
2. O que depende do engine é a memória do contexto de execução (`device_memory_size_v2`)
   mais os pesos (o arquivo .plan). O V do alocador bateu com essa soma em todos os 24
   engines, com diferença entre −0,42 e +0,48 MB.
3. A queda de memória livre gravada pelo benchmark varia (12,58, 25,17, 33,55, 48,23 MB)
   porque o benchmark encaminha as alocações do TensorRT pelo alocador do PyTorch, que
   reserva em segmentos de 2 e 20 MiB. É arredondamento de bloco, não necessidade da rede.

**Conclusão.** O desvio está explicado. V continua como objetivo, medido pelo alocador. A
auditoria da sentinela passa a comparar V com contexto + pesos (tolerância de 5% ou 1 MB),
e a queda de memória livre fica só como referência. Observação para a monografia: a
reserva fixa de 46 MB não entra em V; o V reportado é o custo de memória que depende da
rede, e isso deve estar escrito na definição.

Dados por engine: `auditoria-v-2026-10-06.json`.

| Engine | V alocador (MB) | Contexto + pesos (MB) | Diferença (MB) | Queda no benchmark (MB) | Abrir engine (MB) | 1ª execução (MB) |
|---|---|---|---|---|---|---|
| fila-0006-teste-carga.g012.c0154.a1 | 0,30 | 0,47 | -0,17 | 12,58 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g019.c0118.a1 | 0,43 | 0,55 | -0,11 | 12,58 | 33,55 | 12,58 |
| fila-0006-teste-carga.g007.c0090.a1 | 0,58 | 0,66 | -0,08 | 12,58 | 33,55 | 12,58 |
| fila-0006-teste-carga.g008.c0101.a1 | 0,79 | 0,86 | -0,07 | 12,58 | 33,55 | 12,58 |
| fila-0005-teste-gerac1.g002.c0011.a1 | 0,89 | 1,12 | -0,23 | 12,58 | 33,55 | 12,58 |
| fila-0006-teste-carga.g004.c0050.a1 | 0,97 | 1,21 | -0,24 | 12,58 | 33,55 | 12,58 |
| fila-0006-teste-carga.g004.c0051.a1 | 1,23 | 1,43 | -0,20 | 12,58 | 33,55 | 12,58 |
| fila-0005-teste-gerac1.g001.c0006.a1 | 1,40 | 1,71 | -0,31 | 12,58 | 33,55 | 12,58 |
| fila-0006-teste-carga.g012.c0155.a1 | 1,59 | 1,83 | -0,24 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g014.c0176.a1 | 2,02 | 2,19 | -0,17 | 33,55 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g008.c0049.a1 | 2,43 | 2,61 | -0,19 | 33,55 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g000.c0003.a1 | 2,77 | 3,20 | -0,42 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g005.c0070.a1 | 3,21 | 3,35 | -0,14 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g002.c0034.a1 | 3,93 | 4,08 | -0,16 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g001.c0014.a1 | 4,28 | 4,41 | -0,13 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g002.c0031.a1 | 4,58 | 4,86 | -0,28 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g015.c0191.a1 | 6,71 | 6,88 | -0,18 | 33,55 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g007.c0047.a1 | 7,57 | 7,81 | -0,24 | 33,55 | 33,55 | 12,58 |
| fila-0006-teste-carga.g007.c0094.a1 | 8,93 | 9,19 | -0,26 | 33,55 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g001.c0008.a1 | 12,94 | 12,46 | +0,48 | 25,17 | 33,55 | 12,58 |
| fila-0006-teste-carga.g011.c0138.a1 | 15,22 | 15,39 | -0,17 | 33,55 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g005.c0035.a1 | 16,16 | 15,97 | +0,18 | 48,23 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g010.c0061.a1 | 19,43 | 19,85 | -0,42 | 48,23 | 33,55 | 12,58 |
| fila-0007-test-gens-train.g012.c0074.a1 | 21,51 | 21,60 | -0,09 | 48,23 | 33,55 | 12,58 |
| sentinela | — | 2,79 | — | — | 33,55 | 12,58 |
