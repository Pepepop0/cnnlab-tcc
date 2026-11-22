# cnnlab-tcc

Otimização multiobjetivo de arquitetura e representação numérica de redes neurais
convolucionais (CNNs) para inferência eficiente em GPU.

Este repositório contém o código do Trabalho de Conclusão de Curso de **Pedro Henrique
Alves Antunes** (Bacharelado em Sistemas de Informação, Universidade Federal Rural de
Pernambuco, 2026), orientado pelo Prof. Lucas Fernando da Silva Cambuim.

## O que o trabalho faz

Cada candidata é uma CNN descrita por um genoma, que define a arquitetura, a
representação numérica de cada região da rede (FP32, FP16, BF16 ou INT8, com poda
estruturada) e três hiperparâmetros de treino. A candidata é treinada em PyTorch,
exportada para ONNX e compilada como um *engine* TensorRT. Sobre esse *engine* são
medidos quatro objetivos, todos minimizados:

| Objetivo | Significado |
|---|---|
| `E` | erro de classificação na validação do CIFAR-100, inferido pelo próprio *engine* |
| `L` | latência mediana de uma imagem por vez, em milissegundos |
| `V` | memória de GPU que depende da rede, em megabytes |
| `S` | tamanho do arquivo do *engine*, em megabytes |

A busca conjunta (NSGA-II sobre os quatro objetivos) é comparada, com o mesmo orçamento
de 5,5 horas de GPU por execução, a uma busca monoobjetivo, à amostragem aleatória e a um
fluxo sequencial que busca a arquitetura e só depois a comprime, em oito sementes
pareadas e sob um protocolo definido antes dos experimentos.

**Resultado principal.** A busca conjunta não superou a amostragem aleatória nem o fluxo
sequencial no hipervolume (testes de permutação exatos, p = 0,39 e p = 0,46). A contagem
de operações (MACs) explicou apenas 24% da variância da latência em arquiteturas novas, o
que confirma que o custo precisa ser medido no artefato executado. Todos os métodos
encontraram redes de 1 a 4 MB com 65% a 68% de acurácia no teste oficial.

## Estrutura

```
src/cnnlab/
  tcc/                 implementação do TCC
    science/           genoma, espaço de busca, operadores, NSGA-II, buscas e hipervolume
    model_pipeline/    construção da rede, treino, poda, quantização, ONNX, TensorRT e medição
    operation/         fila de execuções, campanha, fase final, análise estatística e CLI
    storage/           banco SQLite (fonte de verdade), planos congelados e checkpoints
    contracts.py       tipos compartilhados entre as camadas
    demo_*.py          interfaces de acompanhamento (TUI e Streamlit)
    *_wiring.py        ligação entre a camada científica e a de operação
  core/, tui/, worker/, storage/, cli.py, config.py, events.py
                       prova de conceito anterior (algoritmo genético de hiperparâmetros com TUI)
tests/                 testes automatizados (pytest)
tools/campanha/        análise final, redes conhecidas e recortes por limite de custo
tools/piloto/          piloto e regras de decisão fixadas antes do experimento
scripts/               atalhos de execução (PowerShell e .bat para Windows)
docs/tcc/              protocolo, genoma, dados, arquitetura e decisões do TCC
docs/piloto/           auditoria da medida de memória e validação da medida de latência
docs/poc/              especificações da prova de conceito
examples/              configurações de exemplo da prova de conceito
```

## Requisitos

- Python 3.12 ou mais recente (desenvolvido com 3.14)
- GPU NVIDIA com suporte ao TensorRT 11 (os experimentos usaram uma RTX 5070, 12 GB)
- PyTorch com CUDA, instalado conforme a página oficial do PyTorch para a sua GPU
- CIFAR-100 no formato do PyTorch, com o caminho na variável `CNNLAB_TCC_CIFAR_ROOT`

Os testes rápidos rodam sem GPU.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,tcc-trt,tcc-ui]"
```

## Uso

O ponto de entrada é o comando `cnnlab-tcc`. As etapas do experimento, em ordem:

```bash
# 1. Piloto: fixa épocas, repetições de L, ponto de referência e tempo máximo de treino
cnnlab-tcc piloto run ...

# 2. Plano congelado e fila das 32 execuções (4 métodos x 8 sementes)
cnnlab-tcc campaign plan --settings S.json --pilot piloto.json --db tcc.sqlite3 --artifacts artefatos --freeze
cnnlab-tcc campaign enqueue --db tcc.sqlite3 --plan <id-do-plano> --queue-dir fila
cnnlab-tcc fila --dir fila

# 3. Fase final: finalistas, retreino, congelamento e uso único do teste oficial
cnnlab-tcc campaign finalists ...
cnnlab-tcc final prepare | run | artifacts | enqueue-tests | export ...

# 4. Análise
cnnlab-tcc campaign curves --db tcc.sqlite3 --dest analise-hv          # hipervolume (H2, H3)
cnnlab-tcc campaign inference --paired ... --final-results ... --dest inferencia   # testes e H1
cnnlab-tcc campaign h4 --db tcc.sqlite3 --dest analise-h4             # H4
```

Cada subcomando aceita `--help`. Para acompanhar uma execução: `cnnlab-tcc tui` ou
`cnnlab-tcc view`. No Windows, os atalhos de `scripts/windows/` abrem essas interfaces.

Depois da análise, `tools/campanha/analise_final.py` gera os dados das figuras e um
arquivo com todos os números citados no texto, sempre lendo o banco em modo somente
leitura.

## Testes

```bash
pytest -m "not slow"     # testes rápidos, sem GPU nem download de dados
pytest                   # inclui os testes que exigem GPU ou o CIFAR-100
```

## Reprodutibilidade

O experimento foi executado a partir de uma cópia fixa do código, e cada execução grava a
versão do código, o plano congelado e o código de verificação dos dados. O teste oficial
do CIFAR-100 foi usado uma única vez, depois de congelados os finalistas. O código deste
repositório foi reorganizado para publicação sem mudança de comportamento. Os
identificadores do experimento estão no apêndice de reprodutibilidade do texto do TCC.

## Como citar

Veja `CITATION.cff`.

## Licença

MIT. Veja `LICENSE`.
