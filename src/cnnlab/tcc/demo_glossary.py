"""Glossary tab of the read-only page: every term, metric and chart of the other tabs, in plain words.

Static content only: no database access, no project import. ``TERMS`` is the
single source; the tab filters it by a free-text search.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# (category, term, what it is, how to read it)
TERMS: tuple[tuple[str, str, str, str], ...] = (
    # The four objectives
    (
        "Objetivos",
        "E, erro",
        "Fração das 5.000 imagens de validação que o engine classificou errado.",
        "De 0 a 1, menor é melhor. E = 0,78 quer dizer 22% de acerto.",
    ),
    (
        "Objetivos",
        "L, latência",
        "Mediana do tempo de GPU de uma inferência, uma imagem por vez, depois de 200 de aquecimento e 1.000 medidas.",
        "Em milissegundos, menor é melhor. Diferenças abaixo de uns 15% estão dentro do ruído.",
    ),
    (
        "Objetivos",
        "V, memória de vídeo",
        "Pico de memória que a GPU alocou para rodar o engine, acima do que o processo já usava.",
        "Em MB, menor é melhor.",
    ),
    (
        "Objetivos",
        "S, tamanho",
        "Tamanho do arquivo .plan, o engine TensorRT que de fato executa.",
        "Em MB, menor é melhor. Não é a soma dos pesos: inclui a estrutura compilada.",
    ),
    (
        "Objetivos",
        "Minimizar",
        "Os quatro objetivos são minimizados ao mesmo tempo. Não existe nota única que some os quatro.",
        "Em todo gráfico de objetivos, o canto inferior esquerdo é o melhor.",
    ),
    (
        "Objetivos",
        "Preliminar",
        "Marca que o avaliador grava em toda medida do piloto: sem rede sentinela, sem controle de ruído.",
        "Serve para mostrar que o mecanismo funciona, não para concluir nada do TCC.",
    ),
    # Auxiliary metrics
    (
        "Métricas auxiliares",
        "Acurácia",
        "Fração de imagens classificadas corretamente pelo engine. É 1 − E.",
        "Maior é melhor. Não entra na seleção; E já carrega a mesma informação.",
    ),
    (
        "Métricas auxiliares",
        "Top-5",
        "Fração em que a classe certa está entre as 5 que o engine achou mais prováveis.",
        "Maior é melhor. Sempre maior ou igual à acurácia. Só em runs do benchmark v3 em diante.",
    ),
    (
        "Métricas auxiliares",
        "Precisão macro",
        "Para cada classe: das imagens que o engine disse ser dela, quantas eram. Depois, média das 100 classes.",
        "Maior é melhor. Baixa quando o engine chuta muito uma classe só.",
    ),
    (
        "Métricas auxiliares",
        "Recall macro",
        "Para cada classe: das imagens dela, quantas o engine acertou. Depois, média das 100 classes.",
        "Maior é melhor. Na validação do projeto, com 50 imagens por classe, sai igual à acurácia: é esperado.",
    ),
    (
        "Métricas auxiliares",
        "F1 macro",
        "Média harmônica de precisão e recall, calculada classe a classe e depois promediada.",
        "Maior é melhor. Resume as duas; cai quando qualquer uma cai.",
    ),
    (
        "Métricas auxiliares",
        "Acurácia no PyTorch",
        "Acurácia da rede treinada ainda em PyTorch, antes de virar engine.",
        "Deve ficar muito perto da acurácia do engine. Diferença grande indica problema na conversão.",
    ),
    (
        "Métricas auxiliares",
        "Discordâncias",
        "Imagens, das 5.000, em que engine e PyTorch escolheram classes diferentes.",
        "Poucas é normal: a conversão arredonda números. Em FP16 tende a haver um pouco mais.",
    ),
    (
        "Métricas auxiliares",
        "Parâmetros",
        "Número de pesos treináveis da rede.",
        "Rede com mais parâmetros costuma ser maior em S, mas não necessariamente mais lenta em L.",
    ),
    (
        "Métricas auxiliares",
        "GPU (s)",
        "Tempo de GPU ativa gasto numa avaliação: treino, exportação, compilação e medição.",
        "Soma para o orçamento da run. A espera por GPU livre não conta no valor informado pelo avaliador.",
    ),
    # Search
    (
        "Busca evolutiva",
        "Genoma",
        "A receita de uma rede: quantos blocos, que operação em cada um, canais, filtro, precisão, taxa de aprendizado e outros genes.",
        "O algoritmo genético mexe no genoma; o treino por gradiente mexe nos pesos da rede que o genoma descreve.",
    ),
    (
        "Busca evolutiva",
        "Candidata",
        "Um genoma registrado numa run. Nome: g<geração>.c<índice>, por exemplo g001.c0005.",
        "g001 = criada na geração 1; c0005 = sexta candidata da run.",
    ),
    (
        "Busca evolutiva",
        "Tentativa",
        "Uma avaliação de uma candidata. Sufixo .a1, .a2.",
        "Mais de uma tentativa só quando a infraestrutura falhou na anterior.",
    ),
    (
        "Busca evolutiva",
        "Geração",
        "Uma rodada do algoritmo: gera filhos, avalia, escolhe quem sobrevive.",
        "Geração 0 é a população inicial, sorteada.",
    ),
    (
        "Busca evolutiva",
        "População",
        "As candidatas que sobrevivem ao fim de uma geração e servem de pais para a próxima.",
        "Na demo tem 4; no protocolo, 30.",
    ),
    (
        "Busca evolutiva",
        "Dominância",
        "A domina B quando A não é pior em nenhum objetivo e é melhor em pelo menos um.",
        "Se A domina B, não há motivo para escolher B.",
    ),
    (
        "Busca evolutiva",
        "Frente de Pareto",
        "As candidatas que nenhuma outra domina. São os melhores compromissos encontrados.",
        "É o resultado da busca: não um campeão, e sim um conjunto de opções.",
    ),
    (
        "Busca evolutiva",
        "Arquivo de Pareto",
        "A frente calculada sobre tudo que já foi avaliado na run, não só sobre a população atual.",
        "É o que aparece como 'na frente' nas tabelas e gráficos.",
    ),
    (
        "Busca evolutiva",
        "NSGA-II",
        "O algoritmo de seleção usado. Ordena por frentes e, dentro da mesma frente, prefere quem está mais isolado.",
        "Assim a busca melhora os objetivos sem perder variedade.",
    ),
    (
        "Busca evolutiva",
        "Distância de aglomeração",
        "Quão isolada uma candidata está das vizinhas da mesma frente.",
        "Maior é preferida: mantém a frente espalhada.",
    ),
    (
        "Busca evolutiva",
        "Torneio binário",
        "Sorteia duas candidatas e escolhe a melhor para ser mãe ou pai.",
        "Repete até formar todos os pares de pais.",
    ),
    (
        "Busca evolutiva",
        "Cruzamento",
        "Mistura dois pais: cada gene global e cada bloco inteiro vêm de um ou de outro.",
        "Acontece com probabilidade 0,6.",
    ),
    (
        "Busca evolutiva",
        "Mutação",
        "Muda alguns genes de um filho: vizinho em genes ordenados (64 → 96 canais), sorteio em categorias, passo gaussiano em números contínuos (taxa de aprendizado).",
        "Acontece com probabilidade 0,3; cada gene muda com chance 1/n e ao menos um muda. Filho idêntico a um pai sempre é mutado.",
    ),
    (
        "Busca evolutiva",
        "Recorte",
        "Parte do espaço de busca em que alguns genes estão fixos: FP32 (tudo em 32 bits, sem poda) ou FP16 por bloco.",
        "Cada recorte tem hash próprio; runs de recortes diferentes não se misturam.",
    ),
    (
        "Busca evolutiva",
        "Cache",
        "Quando a mesma rede aparece de novo com a mesma receita de avaliação, o resultado é reaproveitado sem gastar GPU.",
        "A receita inclui dados, sementes, políticas de treino, commit, versões de software e o recorte. Mudou qualquer um, avalia de novo.",
    ),
    # Genes
    ("Genes", "Blocos", "Quantos blocos a rede tem, de 2 a 6.", "Mais blocos = rede mais profunda."),
    (
        "Genes",
        "Canais",
        "Quantos filtros cada bloco produz, de 16 a 256.",
        "Mais canais = mais capacidade e mais custo.",
    ),
    ("Genes", "Operação padrão", "Convolução comum.", "A mais cara por canal; a mais simples."),
    (
        "Genes",
        "Operação agrupada",
        "Convolução dividida em 4 grupos independentes.",
        "Cerca de 4 vezes menos contas que a padrão.",
    ),
    (
        "Genes",
        "Operação dw-separável",
        "Um filtro por canal seguido de uma mistura 1×1.",
        "Muito barata; usada em redes para celular.",
    ),
    (
        "Genes",
        "MBConv e3 / e6",
        "Expande os canais 3 ou 6 vezes, filtra e comprime de volta (MobileNetV2).",
        "Boa relação qualidade e custo; e6 é maior que e3.",
    ),
    ("Genes", "Kernel", "Tamanho do filtro espacial: 3, 5 ou 7.", "Maior vê mais contexto e custa mais."),
    (
        "Genes",
        "Convs por bloco",
        "Quantas vezes o bloco repete sua operação, de 1 a 3.",
        "Mais repetições = bloco mais profundo.",
    ),
    ("Genes", "Residual", "Atalho que soma a entrada do bloco à saída.", "Ajuda o treino de redes profundas."),
    (
        "Genes",
        "Downsample",
        "Se e como o bloco reduz a imagem pela metade: max pooling, average pooling ou convolução com passo 2.",
        "No máximo 4 reduções: 32, 16, 8, 4, 2 pixels.",
    ),
    (
        "Genes",
        "Precisão",
        "Número de bits usados no bloco: fp32 (32) ou fp16 (16).",
        "fp16 usa metade da memória; na comparação pareada, V e S caíram à metade, E igual, L sem ganho consistente.",
    ),
    ("Genes", "Poda", "Percentual de canais removidos depois do treino.", "Fixo em 0 nos recortes atuais."),
    # States and operation
    ("Estados e operação", "valid", "Tentativa com os quatro objetivos medidos.", "Entra na seleção."),
    (
        "Estados e operação",
        "invalid",
        "A rede falhou: não treinou, não compilou ou não rodou como pedido. Nenhum valor é inventado.",
        "Fica atrás de todas as válidas na seleção.",
    ),
    (
        "Estados e operação",
        "interrupted",
        "A infraestrutura falhou (GPU ocupada, processo morto). Nada se sabe sobre a candidata.",
        "Não entra na seleção; a candidata é avaliada de novo.",
    ),
    (
        "Estados e operação",
        "Job",
        "Um pedido de execução de uma run, na fila.",
        "Uma run pode ter vários jobs: cancelada e continuada, por exemplo.",
    ),
    (
        "Estados e operação",
        "Lease e heartbeat",
        "O worker 'aluga' o job por 2 minutos e renova a cada 15 s.",
        "'vivo' = o worker está rodando. Se ele morrer, o aluguel vence e outro worker pode assumir.",
    ),
    ("Estados e operação", "Claims", "Quantas vezes um worker assumiu o job.", "Mais de 1 = houve retomada."),
    (
        "Estados e operação",
        "Checkpoint",
        "Estado salvo da busca: população, arquivo, pendentes e sorteador.",
        "Permite retomar depois de uma queda sem refazer nada que já foi confirmado.",
    ),
    (
        "Estados e operação",
        "Orçamento de GPU",
        "Limite de tempo de GPU ativa da run.",
        "A avaliação que cruza o limite termina e conta; depois a run para.",
    ),
    (
        "Estados e operação",
        "Código sujo",
        "Havia arquivo versionado modificado sem commit quando a run foi preparada.",
        "Para resultados apresentados, deve ser 'não'.",
    ),
    (
        "Estados e operação",
        "Versão do sistema",
        "Nome e versão do projeto, estágio, commit exato e rótulo da run.",
        "Diz exatamente com que código cada número foi produzido.",
    ),
    (
        "Estados e operação",
        "Sintético",
        "Resultado fabricado pela fiação de teste, nunca medido.",
        "A página esconde esses números, a menos que você marque a opção na barra lateral.",
    ),
    # Charts
    (
        "Gráficos",
        "Evolução por geração",
        "Melhor e mediana de cada objetivo na população sobrevivente.",
        "E, L, V e S descendo, e acurácia subindo, é bom. A mediana oscila com população pequena.",
    ),
    (
        "Gráficos",
        "Tabela de gerações (aba Operação)",
        "Mínimo de cada objetivo entre as candidatas novas daquela geração.",
        "Pode subir sem a busca piorar: as melhores das gerações anteriores sobrevivem sem ser reavaliadas.",
    ),
    (
        "Gráficos",
        "Frente ao longo da run",
        "Quantas candidatas estão na frente a cada geração, quem entrou e quem saiu.",
        "Frente que ganha membros novos = a busca está achando compromissos novos.",
    ),
    (
        "Gráficos",
        "Todos os pares",
        "Matriz 4×4 cruzando os objetivos dois a dois.",
        "A diagonal é o objetivo contra ele mesmo e forma uma reta. Uma candidata pode parecer ruim num quadro e estar na frente por ser boa em outro.",
    ),
    (
        "Gráficos",
        "Perfil (coordenadas paralelas)",
        "Cada linha é uma candidata passando por E, L, V e S, normalizados de 0 (melhor da run) a 1 (pior).",
        "Linha baixa = boa. Linhas que se cruzam mostram compromissos.",
    ),
    (
        "Gráficos",
        "Equilíbrio",
        "Distância ao canto ideal depois de normalizar os quatro objetivos entre o melhor e o pior da run.",
        "Menor = mais equilibrada. Muda se entrar uma candidata nova que redefina o melhor ou o pior.",
    ),
    (
        "Gráficos",
        "Candidatas parecidas",
        "As de genoma mais próximo de uma candidata escolhida, com o que muda em cada uma.",
        "Mostra o efeito de trocar um gene: por exemplo, fp32 → fp16 no bloco 2.",
    ),
    (
        "Gráficos",
        "Distância de genoma",
        "Fração de genes diferentes entre dois genomas, de 0 (iguais) a 1.",
        "Genes ordenados contam pela distância do passo: 64 → 96 canais pesa menos que 16 → 256.",
    ),
    (
        "Gráficos",
        "Diversidade do genoma",
        "Genomas distintos e distância média por geração, e fração de cada operação e precisão.",
        "Distância média caindo = população convergindo.",
    ),
    ("Gráficos", "Genealogia", "Árvore de pais e filhos, por geração.", "Losango laranja = na frente mais recente."),
    (
        "Gráficos",
        "Curvas de treino",
        "Perda e acurácia de treino época a época.",
        "Curva ainda descendo na última época = 5 épocas é pouco para aquela rede.",
    ),
    (
        "Gráficos",
        "Dispersão da latência",
        "Ponto = mediana (o L). Barra = do percentil 5 ao 95.",
        "Duas candidatas com barras sobrepostas não são claramente diferentes em L.",
    ),
    (
        "Gráficos",
        "Onde o tempo vai",
        "Segundos de treino, exportação, compilação e medição por candidata.",
        "O treino costuma dominar.",
    ),
)


def filtered(query: str) -> list[tuple[str, str, str, str]]:
    """Terms whose name or text contains every word of ``query`` (case-insensitive)."""
    words = [w for w in query.lower().split() if w]
    return [t for t in TERMS if all(w in " ".join(t).lower() for w in words)]


def render_glossary(st: Any) -> None:
    """Draw the glossary tab: a search box and one table per category."""
    import pandas as pd

    st.markdown(
        "Colinha de tudo que aparece nas outras abas. **Resumo:** E, L, V e S são minimizados; "
        "acurácia, top-5, precisão, recall e F1 são maximizados e não entram na seleção."
    )
    query = st.text_input("Buscar termo", key="glossary_query", placeholder="ex.: frente, fp16, recall, lease")
    rows = filtered(query)
    if not rows:
        st.info("Nenhum termo encontrado. Tente uma palavra só.")
        return
    categories: Sequence[str] = list(dict.fromkeys(r[0] for r in rows))
    for category in categories:
        st.subheader(category)
        frame = pd.DataFrame([{"termo": r[1], "o que é": r[2], "como ler": r[3]} for r in rows if r[0] == category])
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            height=min(38 * (len(frame) + 1) + 3, 900),
            column_config={
                "termo": st.column_config.TextColumn(width="small"),
                "o que é": st.column_config.TextColumn(width="large"),
                "como ler": st.column_config.TextColumn(width="large"),
            },
        )
