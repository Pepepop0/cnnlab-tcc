# Modelo de resultados e critérios de relato

**Template sem dados reais · 27/09/2026.** Não contém achados, apenas a estrutura de saída e as verificações a cumprir. [Protocolo](TCC-01-PROTOCOLO.md) · [dados](TCC-04-DADOS.md).

## Resumo executivo futuro

- Instrumento: versões, testes de equivalência PyTorch/ONNX/engine, variabilidade L/V e taxa de engines válidos. Preencher com números e intervalos; publicar falhas.
- Espaço: genoma congelado, opções condicionais admitidas/rejeitadas e seus efeitos demonstrados. Nunca chamar “INT4 acelerado” sem prova do tipo/tática do engine.
- Busca: horas ativas, avaliações válidas únicas, cache/falhas, HV e frentes por seed/método.
- Finalistas: desempenho após retreino, lista congelada e teste oficial; diferenças em relação à busca curta.
- Conclusão: apoio/refutação/indeterminação de H1–H3, descrição de H4, limites de generalização.

## Tabelas obrigatórias

1. **Manifesto experimental:** dataset/split, hardware/software, builder, precisão, política de benchmark, sementes, budget planejado e gasto efetivo.
2. **Piloto/admissão:** por técnica, 3 arquiteturas, build/execução, tática/tipo inspecionado, ΔE/L/V/S, ruído, decisão, motivo e versão do espaço.
3. **Run × método × seed (32 linhas esperadas):** 5,5 h limite, gasto real, status, n propostos, n válidos, n inválidos, cache hits, HV primário e sensibilidade, referência, fora de referência.
4. **Pares H2/H3 (8 linhas por contraste):** HV conjunto, controle, diferença, mesmo bloco/ambiente e custo; teste exato, Holm, tamanho de efeito e IC.
5. **H1 (8 linhas):** campeão mono, solução conjunta escolhida **antes do teste**, Δtop-1 em pp, ΔL ms, ΔV MB, ΔS MB; sucesso/falha conforme regra de 1 pp e todos os custos menores.
6. **Finalistas:** papéis pré-definidos, clones deduplicados, 30 épocas, validação/teste top-1/top-5, custos do engine final, por classe, delta para avaliação curta.
7. **Robustez:** deriva da sentinela, remedições e mudanças entre medições, candidatos fora da referência HV, resultados com ponto sensível e eventuais emendas.

## Figuras obrigatórias

Frentes E×L/E×V/E×S com todos os candidatos relevantes e rótulo de estágio; HV × horas GPU com curvas individuais e resumo entre seeds; HV × válidos únicos; composição do tempo por etapa; correlação de rank entre treinos curtos e longos; latência real versus MACs/parâmetros, marcando família de operação e precisão. Eixos e unidades nunca omitem se o valor vem da busca curta ou de finalista retreinado.

## Regras para interpretação

Não usar indivíduos dentro de uma run como réplicas independentes. Não apresentar só a melhor seed. Não calcular ganho percentual sobre zero nem ocultar indivíduos que falharam build. Reportar evidência contra a hipótese quando houver. Se o teste oficial diferir da validação, discutir sobreajuste da busca, proxy curto e variabilidade do pós-processamento; não substituir finalistas por outros após ver o teste. Se H2/H3 não forem significativas com oito seeds, fornecer efeitos/IC e reconhecer o poder limitado; não mudar alfa, lado ou paridade retrospectivamente. Modelos de H4 são exploratórios e avaliados por arquiteturas não vistas.

Cada tabela/figura final aponta ao `analysis_id`, export do banco e commit/hash do script que a gerou. Espaços vazios neste template não são resultados.
