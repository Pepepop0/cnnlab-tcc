# Latência com CUDA Graph (achado de 06/10/2026, validado em 10/10/2026)

**Problema.** O protocolo define L como a mediana do tempo **de GPU** de uma inferência
no engine, e a proposta mede o engine capturado em CUDA Graph. O benchmark até a versão
v3 chamava `execute_async_v3` direto, sem grafo. Na sentinela, as medianas de L variavam
entre sessões bem acima do limite de 2% do protocolo:

| Condição (sentinela, 1.000 repetições por sessão) | Medianas por sessão (ms) | CV |
|---|---|---|
| Sem grafo, em repouso (06/10) | 0,209 · 0,195 · 0,194 · 0,194 | 3,8% |
| Sem grafo, depois de 60 s de carga (64 °C, 250 W) | 0,206 · 0,266 · 0,229 · 0,213 | 11,8% |
| Sem grafo, em repouso (2ª série) | 0,270 · 0,245 · 0,221 · 0,235 | 8,4% |
| Sem grafo, depois de 45 s de carga | 0,215 · 0,209 · 0,200 · 0,195 | 4,4% |
| **Com grafo, em repouso** | 0,0794 nas 4 sessões | 0,0% |
| **Com grafo, depois de 45 s de carga** | 0,0794 nas 4 sessões | 0,0% |

Sem o grafo, cerca de dois terços dos 0,2 ms eram o custo do host e do driver (WDDM)
para lançar os kernels, e é esse custo que oscila.

**Mudança.** Bench v4 (`tcc-bench/preliminary-v4`, merge `5996a8f`): depois
do aquecimento, uma inferência é capturada num CUDA Graph e repetida no laço medido; E e a
paridade com o PyTorch também passam pelo grafo, então a conferência de paridade valida o
que foi cronometrado. `cuda_graph: true` vai no relatório e na definição do avaliador.

**Validação em 30 engines guardados** (fila, espalhados por toda a faixa de L antiga),
dados em `cuda-graph-validacao-2026-10-10.json`:

- 30 de 30 capturados, nenhuma falha;
- L com grafo / L antigo: mínimo 0,28, mediana 0,39, máximo 0,73;
- Spearman entre L antigo e L com grafo: 0,965 (a ordem quase se mantém, mas não toda);
- L com grafo vai de 0,024 a 0,418 ms (faixa de 17 vezes; a antiga era de cerca de 7).

**Consequências.**

1. Medidas de L de bench v3 e v4 não se misturam. O `bench_version` vai gravado em cada
   tentativa; a análise separa por ele.
2. As conclusões de que FP16, INT8 e poda "não melhoram L em lote 1" (estudos das ondas
   2C e 3) foram tiradas sem grafo e precisam ser refeitas com o bench v4.
3. A captura ainda não foi conferida em engines BF16 e INT8; os estudos refeitos vão
   mostrar. Engine que não capturar não é contornado em silêncio.
