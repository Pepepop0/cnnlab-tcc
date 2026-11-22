# Ferramentas de apoio

Scripts usados no piloto e na análise do experimento. Não fazem parte do pacote
`cnnlab` e não são importados por ele. Todos leem os bancos em modo somente leitura,
e nenhum usa o conjunto de teste oficial.

## `piloto/`: piloto e regras de decisão

| Arquivo | Papel |
|---|---|
| `fidelidade.py` | Treina os primeiros `--n` (20) genomas viáveis do piloto por 5 e por 30 épocas e grava a acurácia de validação. Retoma de onde parou. |
| `ruido.py` | Mede a rede sentinela em sessões separadas, para estimar o ruído da latência. |
| `auditoria_v.py` | Mostra para onde vai a memória da GPU de um *engine*, etapa por etapa. |
| `analisa.py` | Aplica as regras fixadas antes do piloto (épocas da busca, ponto de referência do hipervolume, repetições de L, tempo máximo de treino) e grava `piloto.json` e `PILOTO.md`. |
| `estatistica.py` | Spearman, Pearson e percentis, sem depender do SciPy; testado em `tests/tcc/test_tools_piloto.py`. |
| `trava.py` | Trava de GPU compartilhada, para que só um trabalho use a GPU por vez. |

```bash
python tools/piloto/fidelidade.py --db PILOTO/db/tcc.sqlite3 --run RUN --out PILOTO/fidelidade.json
python tools/piloto/analisa.py --db PILOTO/db/tcc.sqlite3 --run RUN --fidelidade PILOTO/fidelidade.json \
       [--ruido RUIDO.json] --out PILOTO/relatorio
```

## `campanha/`: análise e comparações

| Arquivo | Papel |
|---|---|
| `analise_final.py` | Gera os dados das figuras do texto e um arquivo com todos os números citados, a partir das exportações oficiais da campanha. |
| `referencias.py` | Avalia as duas redes de referência projetadas à mão (baseadas na ResNet e na MobileNetV2) pelo mesmo *pipeline* dos finalistas, só na validação. |
| `referencias_externas.py` | Avalia cinco redes conhecidas para CIFAR (ResNet-20 e -56, MobileNetV2, ShuffleNetV2 e SqueezeNet) pelo mesmo *pipeline*, só na validação. |
| `recortes_restricao.py` | Recortes exploratórios sob limites de L, V e S: melhor acurácia por execução e hipervolume abaixo do limite. |
| `retomar.py` | Continua as execuções que uma pausa deixou pela metade, antes de o executor da fila começar. |
| `ensaio_g9.py` | Ensaio da avaliação final na GPU com um teste falso, para validar o fluxo antes da fase final. |

Os scripts de `campanha/` esperam `PYTHONPATH` apontando para o `src/` da versão do código
usada no experimento.
