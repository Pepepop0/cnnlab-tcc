"""Analysis tab of the read-only page: how a run evolved, what it found and how candidates compare.

Everything here reads through the read model (``mode=ro`` + ``query_only``); the
module opens no connection of its own and cannot write. The data layer (plain
functions over read-model rows) has no streamlit import, so tests exercise it
directly; ``render_analysis`` draws it.

All four objectives are minimized. Accuracy is ``1 - E``. Top-5 and the macro
precision, recall and F1 are auxiliary: they exist only for attempts measured
by benchmark v3 or later and never enter the fitness.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cnnlab.tcc.storage.readmodel import open_read_model

OBJECTIVES: tuple[tuple[str, str, str], ...] = (
    ("e", "E", "erro top-1"),
    ("l_ms", "L", "latência (ms)"),
    ("v_mb", "V", "memória (MB)"),
    ("s_mb", "S", "tamanho (MB)"),
)
LOG_OBJECTIVES = ("v_mb", "s_mb")  # spread over an order of magnitude
BLOCK_GENES = ("channels", "op", "kernel", "n_convs", "residual", "downsample", "precision")
GLOBAL_GENES = ("n_blocks", "stem_channels", "activation", "head")
ORDINAL_DOMAINS: dict[str, tuple[Any, ...]] = {
    "n_blocks": (2, 3, 4, 5, 6),
    "stem_channels": (16, 32, 64),
    "channels": (16, 32, 48, 64, 96, 128, 192, 256),
    "kernel": (3, 5, 7),
    "n_convs": (1, 2, 3),
}
OP_NAMES = {
    "standard": "padrão",
    "grouped_g4": "agrupada",
    "dw_separable": "dw-separável",
    "mbconv_e3": "MBConv e3",
    "mbconv_e6": "MBConv e6",
}

# Colors from the validated palette (dataviz check: light and dark surfaces).
FRONT_COLORS = ("#3987e5", "#d95926")  # off front, in front
OFF_FRONT, IN_FRONT = "fora da frente", "na frente"
INK = {"light": "#0b0b0b", "dark": "#e8eef6"}  # direct labels wear text ink, never a series color
GENERATION_COLORS = ("#9ec5f4", "#3987e5", "#1c5cab", "#104281", "#0d366b")

HELP: dict[str, str] = {
    "acuracia": "Fração de imagens de validação classificadas corretamente pelo engine: 1 − E. Maior é melhor.",
    "top5": "Fração em que a classe certa está entre as 5 mais prováveis. Só em runs medidas com o benchmark v3 ou posterior.",
    "precisao_macro": "Média, sobre as 100 classes, de: das imagens que o engine disse ser da classe, quantas eram. Maior é melhor.",
    "recall_macro": "Média, sobre as 100 classes, de: das imagens da classe, quantas o engine acertou. Maior é melhor.",
    "f1_macro": "Média harmônica de precisão e recall, calculada por classe e depois promediada. Maior é melhor.",
    "acuracia_pytorch": "Acurácia da rede em PyTorch, antes da conversão para engine. Serve para ver quanto a conversão mudou.",
    "discordancias": "Imagens, de 5.000, em que o engine e o PyTorch escolheram classes diferentes.",
    "parametros": "Número de pesos treináveis da rede.",
    "E": "Erro top-1 do engine, de 0 a 1. Menor é melhor.",
    "L_ms": "Latência: mediana do tempo de GPU de uma inferência, lote 1. Menor é melhor.",
    "V_MB": "Pico de memória de vídeo da inferência, em MB. Menor é melhor.",
    "S_MB": "Tamanho do arquivo do engine, em MB. Menor é melhor.",
    "distancia_equilibrio": "Distância à origem depois de normalizar cada objetivo entre o melhor e o pior desta run. Menor = mais equilibrada nos quatro.",
    "distancia_genoma": "Fração de genes diferentes: 0 = mesmo genoma, 1 = nada em comum. Genes ordinais contam proporcionalmente ao passo.",
    "na_frente": "Se a candidata está na frente de Pareto mais recente: nenhuma outra é melhor nos quatro objetivos ao mesmo tempo.",
}


@dataclass(frozen=True)
class AnalysisView:
    run_id: str
    details: list[dict[str, Any]]
    genomes: dict[str, dict[str, Any]]
    lineage: dict[str, dict[str, Any]]
    populations: list[dict[str, Any]]
    fronts: list[dict[str, Any]]

    @property
    def front(self) -> set[str]:
        return set(self.fronts[-1]["candidate_ids"]) if self.fronts else set()

    @property
    def has_macro_metrics(self) -> bool:
        return any(d["macro_f1"] is not None for d in self.details)


def load_analysis(db_path: str, run_id: str) -> AnalysisView:
    """Everything the tab needs, read in one pass through the read model."""
    with open_read_model(db_path) as rm:
        details = [d for d in rm.evaluation_details(run_id) if not d["synthetic"]]
        genomes = {g["candidate_id"]: g for g in rm.genomes(run_id)}
        lineage = {row["candidate_id"]: row for row in rm.lineage(run_id)}
        populations = rm.populations(run_id)
        fronts = rm.fronts(run_id)
    return AnalysisView(run_id, details, genomes, lineage, populations, fronts)


def short(candidate_id: str) -> str:
    """``<run>.g001.c0005`` -> ``g001.c0005``."""
    return ".".join(candidate_id.split(".")[-2:])


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100 * value, 2)


# -- data layer --------------------------------------------------------------------------


def quality_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """One row per measured candidate: classification quality, in percent."""
    front = view.front
    return [
        {
            "candidata": short(d["candidate_id"]),
            "geracao": d["generation"],
            "na_frente": d["candidate_id"] in front,
            "acuracia": _pct(d["accuracy"]),
            "top5": _pct(d["top5_accuracy"]),
            "precisao_macro": _pct(d["macro_precision"]),
            "recall_macro": _pct(d["macro_recall"]),
            "f1_macro": _pct(d["macro_f1"]),
            "acuracia_pytorch": _pct(d["torch_accuracy"]),
            "discordancias": d["parity_disagreements"],
            "parametros": d["parameters"],
        }
        for d in view.details
    ]


def generation_populations(view: AnalysisView) -> dict[int, list[str]]:
    """Population after selection at the end of each generation (last checkpoint of that generation)."""
    last: dict[int, int] = {}
    for row in view.populations:
        last[row["generation"]] = max(last.get(row["generation"], -1), row["checkpoint_id"])
    out: dict[int, list[str]] = {}
    for row in view.populations:
        if last[row["generation"]] == row["checkpoint_id"]:
            out.setdefault(row["generation"], []).append(row["candidate_id"])
    return out


def evolution_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """Best and median of each objective in the surviving population, generation by generation."""
    measured = {d["candidate_id"]: d for d in view.details}
    rows = []
    for generation, members in sorted(generation_populations(view).items()):
        scored = [measured[c] for c in members if c in measured]
        if not scored:
            continue
        for key, letter, name in OBJECTIVES:
            values = [s[key] for s in scored]
            rows.append(
                {"geracao": generation, "objetivo": f"{letter}, {name}", "estatistica": "melhor", "valor": min(values)}
            )
            rows.append(
                {
                    "geracao": generation,
                    "objetivo": f"{letter}, {name}",
                    "estatistica": "mediana",
                    "valor": statistics.median(values),
                }
            )
        rows.append(
            {
                "geracao": generation,
                "objetivo": "Acurácia (%)",
                "estatistica": "melhor",
                "valor": 100 * max(s["accuracy"] for s in scored),
            }
        )
        rows.append(
            {
                "geracao": generation,
                "objetivo": "Acurácia (%)",
                "estatistica": "mediana",
                "valor": 100 * statistics.median(s["accuracy"] for s in scored),
            }
        )
    return rows


def front_history_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """For each recorded front: size, who entered, who left, and the best value of each objective in it."""
    measured = {d["candidate_id"]: d for d in view.details}
    rows, previous = [], set()
    for index, snap in enumerate(view.fronts, start=1):
        members = set(snap["candidate_ids"])
        scored = [measured[c] for c in members if c in measured]
        row = {
            "retrato": index,
            "gpu_ativa_s": round(snap["active_gpu_seconds"], 1),
            "membros": len(members),
            "entraram": " ".join(sorted(short(c) for c in members - previous)) or "-",
            "sairam": " ".join(sorted(short(c) for c in previous - members)) or "-",
        }
        for key, letter, _ in OBJECTIVES:
            row[f"melhor_{letter}"] = min((s[key] for s in scored), default=None)
        rows.append(row)
        previous = members
    return rows


def _normalized(view: AnalysisView) -> dict[str, dict[str, float]]:
    """Each objective scaled to [0, 1] between the best and the worst measured value of the run."""
    out: dict[str, dict[str, float]] = {d["candidate_id"]: {} for d in view.details}
    for key, _, _ in OBJECTIVES:
        values = [d[key] for d in view.details]
        low, high = min(values), max(values)
        for d in view.details:
            out[d["candidate_id"]][key] = 0.0 if high == low else (d[key] - low) / (high - low)
    return out


def ranking_rows(view: AnalysisView, order_by: str) -> list[dict[str, Any]]:
    """Candidates ordered by one objective, accuracy, or balance (distance to the ideal corner)."""
    norm = _normalized(view)
    front = view.front
    rows = []
    for d in view.details:
        n = norm[d["candidate_id"]]
        rows.append(
            {
                "candidata": short(d["candidate_id"]),
                "geracao": d["generation"],
                "na_frente": d["candidate_id"] in front,
                "acuracia": _pct(d["accuracy"]),
                "E": d["e"],
                "L_ms": d["l_ms"],
                "V_MB": d["v_mb"],
                "S_MB": d["s_mb"],
                "distancia_equilibrio": round(math.sqrt(sum(v * v for v in n.values())), 4),
                "precisao_por_bloco": " ".join(str(b.get("precision")) for b in _blocks(view, d["candidate_id"])),
            }
        )
    keys = {
        "equilibrio": "distancia_equilibrio",
        "E": "E",
        "L": "L_ms",
        "V": "V_MB",
        "S": "S_MB",
        "acuracia": "acuracia",
    }
    key = keys[order_by]
    rows.sort(key=lambda r: (-r[key] if key == "acuracia" else r[key], r["candidata"]))
    for position, row in enumerate(rows, start=1):
        row["posicao"] = position
    return rows


def _blocks(view: AnalysisView, candidate_id: str) -> list[dict[str, Any]]:
    genome = view.genomes.get(candidate_id) or {}
    return list(genome.get("blocks") or [])


def _gene_distance(name: str, a: Any, b: Any) -> float:
    if a == b:
        return 0.0
    domain = ORDINAL_DOMAINS.get(name)
    if domain and a in domain and b in domain:
        return abs(domain.index(a) - domain.index(b)) / (len(domain) - 1)
    return 1.0


def genome_distance(view: AnalysisView, a: str, b: str) -> float | None:
    """Fraction of differing genes of two expressed genomes; blocks are compared slot by slot."""
    ga, gb = view.genomes.get(a), view.genomes.get(b)
    if not ga or not gb or not ga.get("blocks_recorded") or not gb.get("blocks_recorded"):
        return None
    total, diff = 0, 0.0
    for name in GLOBAL_GENES:
        total += 1
        diff += _gene_distance(name, ga.get(name), gb.get(name))
    blocks_a, blocks_b = ga.get("blocks") or [], gb.get("blocks") or []
    for slot in range(max(len(blocks_a), len(blocks_b))):
        for name in BLOCK_GENES:
            total += 1
            if slot >= len(blocks_a) or slot >= len(blocks_b):
                diff += 1.0  # a block only one of them has
            else:
                diff += _gene_distance(name, blocks_a[slot].get(name), blocks_b[slot].get(name))
    return round(diff / total, 4) if total else None


def similar_rows(view: AnalysisView, candidate_id: str, limit: int = 6) -> list[dict[str, Any]]:
    """The measured candidates closest in genome to ``candidate_id``, with how each objective compares."""
    measured = {d["candidate_id"]: d for d in view.details}
    base = measured.get(candidate_id)
    rows = []
    for other, d in measured.items():
        if other == candidate_id:
            continue
        distance = genome_distance(view, candidate_id, other)
        if distance is None:
            continue
        row = {
            "candidata": short(other),
            "geracao": d["generation"],
            "distancia_genoma": distance,
            "na_frente": other in view.front,
            "acuracia": _pct(d["accuracy"]),
            "E": d["e"],
            "L_ms": d["l_ms"],
            "V_MB": d["v_mb"],
            "S_MB": d["s_mb"],
        }
        if base is not None:
            row["diferencas"] = _describe_difference(view, candidate_id, other)
        rows.append(row)
    rows.sort(key=lambda r: (r["distancia_genoma"], r["candidata"]))
    return rows[:limit]


def _describe_difference(view: AnalysisView, a: str, b: str) -> str:
    ga, gb = view.genomes[a], view.genomes[b]
    parts = []
    for name in GLOBAL_GENES:
        if ga.get(name) != gb.get(name):
            parts.append(f"{name}: {ga.get(name)} → {gb.get(name)}")
    blocks_a, blocks_b = ga.get("blocks") or [], gb.get("blocks") or []
    for slot in range(min(len(blocks_a), len(blocks_b))):
        changed = [
            f"{n} {blocks_a[slot].get(n)}→{blocks_b[slot].get(n)}"
            for n in BLOCK_GENES
            if blocks_a[slot].get(n) != blocks_b[slot].get(n)
        ]
        if changed:
            parts.append(f"bloco {slot + 1}: " + ", ".join(changed))
    return "; ".join(parts[:4]) + (" …" if len(parts) > 4 else "") if parts else "mesmo genoma"


def diversity_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """Share of each block operation and each precision among the candidates created in a generation."""
    rows = []
    by_generation: dict[int, list[str]] = {}
    for candidate_id, genome in view.genomes.items():
        by_generation.setdefault(genome["generation"], []).append(candidate_id)
    for generation, members in sorted(by_generation.items()):
        for gene, label in (("op", "operação"), ("precision", "precisão")):
            counts = Counter(str(b.get(gene)) for c in members for b in _blocks(view, c))
            total = sum(counts.values()) or 1
            for value, count in sorted(counts.items()):
                shown = OP_NAMES.get(value, value) if gene == "op" else value
                rows.append(
                    {"geracao": generation, "gene": label, "valor": shown, "fracao": count / total, "blocos": count}
                )
    return rows


def diversity_summary(view: AnalysisView) -> list[dict[str, Any]]:
    """Candidates, distinct genomes and mean pairwise genome distance per generation."""
    by_generation: dict[int, list[str]] = {}
    for candidate_id, genome in view.genomes.items():
        by_generation.setdefault(genome["generation"], []).append(candidate_id)
    rows = []
    for generation, members in sorted(by_generation.items()):
        distances = [genome_distance(view, a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
        distances = [d for d in distances if d is not None]
        rows.append(
            {
                "geracao": generation,
                "candidatas": len(members),
                "genomas_distintos": len({view.genomes[c]["phenotype_hash"] for c in members}),
                "distancia_media": round(statistics.fmean(distances), 3) if distances else None,
            }
        )
    return rows


def lineage_edges(view: AnalysisView) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Nodes (candidate, generation, slot) and parent->child edges for the family tree."""
    measured = {d["candidate_id"]: d for d in view.details}
    front = view.front
    slots: dict[int, int] = {}
    nodes = []
    position: dict[str, tuple[int, int]] = {}
    for candidate_id, row in view.lineage.items():
        generation = row["generation"] or 0
        slot = slots.get(generation, 0)
        slots[generation] = slot + 1
        position[candidate_id] = (generation, slot)
        d = measured.get(candidate_id)
        nodes.append(
            {
                "candidata": short(candidate_id),
                "geracao": generation,
                "posicao": slot,
                "frente": IN_FRONT if candidate_id in front else OFF_FRONT,
                "E": d["e"] if d else None,
                "acuracia": _pct(d["accuracy"]) if d else None,
                "operadores": _operators(row),
            }
        )
    edges = []
    for candidate_id, row in view.lineage.items():
        for parent in dict.fromkeys(row["parents"]):
            if parent in position:
                (g0, s0), (g1, s1) = position[parent], position[candidate_id]
                edges.append({"de": short(parent), "para": short(candidate_id), "g0": g0, "s0": s0, "g1": g1, "s1": s1})
    return nodes, edges


def _operators(row: Mapping[str, Any]) -> str:
    if row.get("origin") != "offspring":
        return "população inicial" if row.get("origin") == "initial" else str(row.get("origin") or "-")
    used = [name for name, on in (("cruzamento", row.get("crossover")), ("mutação", row.get("mutated"))) if on]
    return " e ".join(used) or "cópia"


def training_rows(view: AnalysisView, candidates: Sequence[str]) -> list[dict[str, Any]]:
    """Loss and training accuracy per epoch of the chosen candidates."""
    wanted = set(candidates)
    rows = []
    for d in view.details:
        name = short(d["candidate_id"])
        if name not in wanted:
            continue
        for epoch, (loss, acc) in enumerate(zip(d["epoch_loss"], d["epoch_train_acc"]), start=1):
            rows.append({"candidata": name, "epoca": epoch, "perda": loss, "acuracia_treino": 100 * acc})
    return rows


def latency_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """Median latency with the 5th and 95th percentiles of the 1,000 timed inferences."""
    return [
        {
            "candidata": short(d["candidate_id"]),
            "frente": IN_FRONT if d["candidate_id"] in view.front else OFF_FRONT,
            "mediana": d["l_ms"],
            "p5": d["l_p5_ms"],
            "p95": d["l_p95_ms"],
        }
        for d in view.details
        if d["l_p5_ms"] is not None
    ]


def time_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """Seconds spent in each stage of each evaluation."""
    stages = {"train": "treino", "export": "exportação", "trt_build": "compilação", "benchmark": "medição"}
    return [
        {"candidata": short(d["candidate_id"]), "etapa": label, "segundos": d["durations"][key]}
        for d in view.details
        for key, label in stages.items()
        if key in d["durations"]
    ]


def objective_rows(view: AnalysisView) -> list[dict[str, Any]]:
    """Measured candidates with their four objectives, for the scatter matrix and parallel coordinates."""
    front = view.front
    norm = _normalized(view)
    rows = []
    for d in view.details:
        row = {
            "candidata": short(d["candidate_id"]),
            "geracao": d["generation"],
            "frente": IN_FRONT if d["candidate_id"] in front else OFF_FRONT,
            "E": d["e"],
            "L_ms": d["l_ms"],
            "V_MB": d["v_mb"],
            "S_MB": d["s_mb"],
            "acuracia": _pct(d["accuracy"]),
        }
        for key, letter, _ in OBJECTIVES:
            row[f"norm_{letter}"] = norm[d["candidate_id"]][key]
        rows.append(row)
    return rows


# -- page ---------------------------------------------------------------------------------


def _table(st: Any, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    import pandas as pd

    frame = pd.DataFrame(list(rows), columns=list(columns) if columns else None)
    config = {name: st.column_config.Column(help=HELP[name]) for name in frame.columns if name in HELP}
    height = min(35 * (len(frame) + 1) + 3, 420)
    st.dataframe(frame, hide_index=True, height=height, column_config=config)


def render_analysis(st: Any, db_path: str, run_ids: Sequence[str]) -> None:
    """Draw the analysis tab for one run chosen here (independent of the operation tab)."""
    import altair as alt
    import pandas as pd

    try:
        theme = st.context.theme.type
    except Exception:  # older streamlit or no browser context
        theme = None
    ink = INK.get(theme or "light", INK["light"])
    front_color = alt.Color(
        "frente:N", title=None, scale=alt.Scale(domain=[OFF_FRONT, IN_FRONT], range=list(FRONT_COLORS))
    )
    front_shape = alt.Shape(
        "frente:N", title=None, scale=alt.Scale(domain=[OFF_FRONT, IN_FRONT], range=["circle", "diamond"])
    )

    run_id = st.selectbox("Run analisada", list(run_ids), index=len(run_ids) - 1, key="analysis_run_id")
    view = load_analysis(db_path, run_id)
    if not view.details:
        st.info("Esta run ainda não tem candidata medida. A análise aparece quando a primeira avaliação terminar.")
        return

    st.markdown(
        "Todos os objetivos são **minimizados**: menor é melhor em E, L, V e S. A acurácia é `1 − E`. "
        "Passe o mouse no nome de uma coluna ou num ponto para ver o que ele significa."
    )

    # Summary tiles: the best candidate on each axis.
    best_acc = max(view.details, key=lambda d: d["accuracy"])
    cols = st.columns(5)
    cols[0].metric("Melhor acurácia", f"{100 * best_acc['accuracy']:.2f}%", help=short(best_acc["candidate_id"]))
    for col, (key, letter, name) in zip(cols[1:], OBJECTIVES[1:] + (("gpu_seconds", "GPU", "s"),)):
        best = min(view.details, key=lambda d: d[key])
        unit = {"l_ms": " ms", "v_mb": " MB", "s_mb": " MB", "gpu_seconds": " s"}[key]
        title = {
            "l_ms": "Menor latência",
            "v_mb": "Menor memória",
            "s_mb": "Menor tamanho",
            "gpu_seconds": "Avaliação mais rápida",
        }[key]
        col.metric(title, f"{best[key]:.3f}{unit}", help=short(best["candidate_id"]))
    st.caption(
        f"{len(view.details)} candidatas medidas, {len(view.front)} na frente mais recente. "
        "O nome da candidata aparece ao passar o mouse no (?) de cada número."
    )

    # 1. Classification quality.
    st.subheader("Qualidade da classificação")
    if not view.has_macro_metrics:
        st.info(
            "Top-5, precisão, recall e F1 só existem para runs medidas depois de 29/09/2026 (benchmark v3). "
            "Esta run mostra a acurácia, que vem de E."
        )
    quality = quality_rows(view)
    columns = [
        "candidata",
        "geracao",
        "na_frente",
        "acuracia",
        "top5",
        "precisao_macro",
        "recall_macro",
        "f1_macro",
        "acuracia_pytorch",
        "discordancias",
        "parametros",
    ]
    if not view.has_macro_metrics:
        columns = [c for c in columns if c not in ("top5", "precisao_macro", "recall_macro", "f1_macro")]
    _table(st, quality, columns)

    # 2. Evolution per generation.
    st.subheader("Evolução por geração")
    st.caption(
        "Melhor e mediana da população sobrevivente ao fim de cada geração. Para E, L, V e S, descer é "
        "melhorar; para a acurácia, subir. Com população pequena, a mediana oscila bastante."
    )
    evo = pd.DataFrame(evolution_rows(view))
    if not evo.empty:
        chart = (
            alt.Chart(evo)
            .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2)
            .encode(
                x=alt.X("geracao:O", title="geração"),
                y=alt.Y("valor:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "estatistica:N",
                    title=None,
                    scale=alt.Scale(domain=["melhor", "mediana"], range=list(FRONT_COLORS[::-1])),
                ),
                tooltip=["objetivo", "geracao", "estatistica", alt.Tooltip("valor:Q", format=".4f")],
            )
            .properties(width=300, height=180)
            .facet(facet=alt.Facet("objetivo:N", title=None), columns=3)
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart)

    # 3. Front over time.
    st.subheader("Frente de Pareto ao longo da run")
    st.caption(
        "Cada retrato é a frente gravada ao fim de uma geração: quantas candidatas não são dominadas, "
        "quem entrou, quem saiu e o melhor valor de cada objetivo dentro dela."
    )
    if view.fronts:
        _table(st, front_history_rows(view))
    else:
        st.caption("Nenhuma frente gravada ainda.")

    # 4. All pairs of objectives.
    obj = pd.DataFrame(objective_rows(view))
    st.subheader("Todos os pares de objetivos")
    st.caption(
        "Cada quadro cruza dois objetivos. O canto inferior esquerdo é o melhor. Uma candidata dominada "
        "num quadro pode estar na frente por ser melhor num objetivo que aquele quadro não mostra."
    )
    fields = ["E", "L_ms", "V_MB", "S_MB"]
    matrix = (
        alt.Chart(obj)
        .mark_point(filled=True, size=90, opacity=0.9)
        .encode(
            x=alt.X(alt.repeat("column"), type="quantitative", scale=alt.Scale(zero=False)),
            y=alt.Y(alt.repeat("row"), type="quantitative", scale=alt.Scale(zero=False)),
            color=front_color,
            shape=front_shape,
            tooltip=[
                "candidata",
                "geracao",
                "frente",
                alt.Tooltip("acuracia:Q", title="acurácia (%)"),
                alt.Tooltip("E:Q", format=".4f"),
                alt.Tooltip("L_ms:Q", format=".4f"),
                alt.Tooltip("V_MB:Q", format=".3f"),
                alt.Tooltip("S_MB:Q", format=".3f"),
            ],
        )
        .properties(width=230, height=170)
        .repeat(row=fields, column=fields)
    )
    st.altair_chart(matrix)

    # 5. Parallel coordinates.
    st.subheader("Perfil de cada candidata")
    st.caption(
        "Coordenadas paralelas: cada linha é uma candidata passando pelos quatro objetivos, normalizados "
        "entre o melhor (0) e o pior (1) desta run. Linhas baixas são boas. Linhas que se cruzam mostram "
        "compromissos: melhor num objetivo, pior em outro."
    )
    long = obj.melt(
        id_vars=["candidata", "frente", "geracao"],
        value_vars=[f"norm_{l}" for _, l, _ in OBJECTIVES],
        var_name="objetivo",
        value_name="valor_normalizado",
    )
    long["objetivo"] = long["objetivo"].str.replace("norm_", "", regex=False)
    parallel = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, opacity=0.75)
        .encode(
            x=alt.X("objetivo:N", sort=["E", "L", "V", "S"], title=None),
            y=alt.Y("valor_normalizado:Q", title="0 = melhor da run, 1 = pior", scale=alt.Scale(domain=[0, 1])),
            detail="candidata:N",
            color=front_color,
            tooltip=["candidata", "frente", "objetivo", alt.Tooltip("valor_normalizado:Q", format=".2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(parallel, width="stretch")

    # 6. Best candidates.
    st.subheader("Melhores candidatas")
    choice = st.radio(
        "Ordenar por", ["equilíbrio", "acurácia", "E", "L", "V", "S"], horizontal=True, key="analysis_rank_by"
    )
    order = {"equilíbrio": "equilibrio", "acurácia": "acuracia"}.get(choice, choice)
    st.caption(
        "Equilíbrio: distância ao canto ideal depois de normalizar os quatro objetivos; menor é mais "
        "equilibrada. É o critério de 'ponto de equilíbrio' do protocolo, aqui com a escala desta run."
    )
    _table(
        st,
        ranking_rows(view, order),
        [
            "posicao",
            "candidata",
            "geracao",
            "na_frente",
            "acuracia",
            "E",
            "L_ms",
            "V_MB",
            "S_MB",
            "distancia_equilibrio",
            "precisao_por_bloco",
        ],
    )

    # 7. Similar candidates.
    st.subheader("Candidatas parecidas")
    names = [short(d["candidate_id"]) for d in view.details]
    by_short = {short(d["candidate_id"]): d["candidate_id"] for d in view.details}
    default = names.index(short(best_acc["candidate_id"]))
    chosen = st.selectbox("Candidata de referência", names, index=default, key="analysis_similar_to")
    ref = next(d for d in view.details if d["candidate_id"] == by_short[chosen])
    st.caption(
        f"Referência {chosen}: acurácia {100 * ref['accuracy']:.2f}%, L {ref['l_ms']:.4f} ms, "
        f"V {ref['v_mb']:.3f} MB, S {ref['s_mb']:.3f} MB. Abaixo, as de genoma mais próximo e o que muda "
        "em cada uma: mostra que genes custam ou poupam."
    )
    similar = similar_rows(view, by_short[chosen])
    if similar:
        _table(
            st,
            similar,
            [
                "candidata",
                "geracao",
                "distancia_genoma",
                "na_frente",
                "acuracia",
                "E",
                "L_ms",
                "V_MB",
                "S_MB",
                "diferencas",
            ],
        )
    else:
        st.caption("Sem outra candidata com genoma registrado para comparar.")

    # 8. Genome diversity.
    st.subheader("Diversidade do genoma")
    st.caption(
        "Quanto a população varia de uma geração para a outra. Distância média perto de 0 = população "
        "convergindo para genomas parecidos; perto de 1 = ainda explorando."
    )
    _table(st, diversity_summary(view))
    div = pd.DataFrame(diversity_rows(view))
    if not div.empty:
        for gene in ("operação", "precisão"):
            part = div[div["gene"] == gene]
            bars = (
                alt.Chart(part)
                .mark_bar()
                .encode(
                    x=alt.X("geracao:O", title="geração"),
                    y=alt.Y(
                        "fracao:Q", title=f"fração dos blocos ({gene})", axis=alt.Axis(format="%"), stack="normalize"
                    ),
                    color=alt.Color("valor:N", title=gene),
                    tooltip=["geracao", "valor", "blocos", alt.Tooltip("fracao:Q", format=".0%")],
                )
                .properties(height=220, width=320)
            )
            st.altair_chart(bars)

    # 9. Family tree.
    st.subheader("Genealogia")
    st.caption(
        "Cada ponto é uma candidata, organizada por geração. As linhas ligam pais a filhos. Losangos "
        "laranja estão na frente mais recente."
    )
    nodes, edges = lineage_edges(view)
    node_frame = pd.DataFrame(nodes)
    last_generation = int(node_frame["geracao"].max())
    gen_axis = alt.X(
        "geracao:Q",
        title="geração",
        axis=alt.Axis(tickMinStep=1, format="d"),
        scale=alt.Scale(domain=[-0.15, last_generation + 0.45]),
    )
    layers = []
    if edges:
        layers.append(
            alt.Chart(pd.DataFrame(edges))
            .mark_rule(strokeWidth=1.5, opacity=0.5, color="#8a99ae")
            .encode(
                x=alt.X(
                    "g0:Q",
                    title="geração",
                    axis=alt.Axis(tickMinStep=1, format="d"),
                    scale=alt.Scale(domain=[-0.15, last_generation + 0.45]),
                ),
                y=alt.Y("s0:Q", title=None, axis=None),
                x2="g1:Q",
                y2="s1:Q",
                tooltip=[alt.Tooltip("de:N", title="pai"), alt.Tooltip("para:N", title="filho")],
            )
        )
    layers.append(
        alt.Chart(node_frame)
        .mark_point(filled=True, size=160, opacity=1)
        .encode(
            x=gen_axis,
            y=alt.Y("posicao:Q", title=None, axis=None),
            color=front_color,
            shape=front_shape,
            tooltip=[
                "candidata",
                "geracao",
                "operadores",
                alt.Tooltip("acuracia:Q", title="acurácia (%)"),
                alt.Tooltip("E:Q", format=".4f"),
            ],
        )
    )
    layers.append(
        alt.Chart(node_frame)
        .mark_text(dx=10, dy=-10, align="left", fontSize=12, color=ink)
        .encode(x="geracao:Q", y="posicao:Q", text="candidata:N")
    )
    st.altair_chart(
        alt.layer(*layers).properties(height=60 * (1 + node_frame["posicao"].max())).configure_legend(orient="bottom"),
        width="stretch",
    )

    # 10. Training curves.
    st.subheader("Curvas de treino")
    front_names = [short(c) for c in sorted(view.front)] or names[:4]
    picked = st.multiselect(
        "Candidatas", names, default=[n for n in front_names if n in names][:6], key="analysis_training_pick"
    )
    st.caption(
        "Perda e acurácia no conjunto de treino, época a época. Curvas ainda descendo no fim indicam que "
        "5 épocas é pouco para aquela rede; o piloto do protocolo decide entre 5 e 8."
    )
    train = pd.DataFrame(training_rows(view, picked))
    if not train.empty:
        for field, title in (("perda", "perda de treino"), ("acuracia_treino", "acurácia de treino (%)")):
            st.altair_chart(
                alt.Chart(train)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("epoca:O", title="época"),
                    y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False)),
                    color=alt.Color("candidata:N", title=None),
                    tooltip=["candidata", "epoca", alt.Tooltip(f"{field}:Q", format=".4f")],
                )
                .properties(height=240, width=460)
            )

    # 11. Latency spread.
    lat = pd.DataFrame(latency_rows(view))
    if not lat.empty:
        st.subheader("Dispersão da latência")
        st.caption(
            "Ponto: mediana das 1.000 inferências (é o L). Barra: do percentil 5 ao 95. Barras longas "
            "indicam medida ruidosa; diferenças menores que a barra não devem ser levadas a sério."
        )
        base = alt.Chart(lat).encode(y=alt.Y("candidata:N", sort="x", title=None))
        st.altair_chart(
            (
                base.mark_rule(strokeWidth=2, color="#8a99ae").encode(
                    x=alt.X("p5:Q", title="latência (ms)"), x2="p95:Q"
                )
                + base.mark_point(filled=True, size=90).encode(
                    x="mediana:Q",
                    color=front_color,
                    tooltip=[
                        "candidata",
                        alt.Tooltip("mediana:Q", format=".4f"),
                        alt.Tooltip("p5:Q", format=".4f"),
                        alt.Tooltip("p95:Q", format=".4f"),
                    ],
                )
            ).properties(height=24 * len(lat) + 40),
            width="stretch",
        )

    # 12. Where the time goes.
    spent = pd.DataFrame(time_rows(view))
    if not spent.empty:
        st.subheader("Onde o tempo de GPU vai")
        st.caption(
            "Segundos de cada etapa por candidata. O treino costuma dominar; compilação alta indica "
            "rede com muitas camadas para o TensorRT otimizar."
        )
        st.altair_chart(
            alt.Chart(spent)
            .mark_bar()
            .encode(
                y=alt.Y("candidata:N", title=None),
                x=alt.X("segundos:Q", title="segundos", stack="zero"),
                color=alt.Color("etapa:N", title=None, sort=["treino", "exportação", "compilação", "medição"]),
                tooltip=["candidata", "etapa", alt.Tooltip("segundos:Q", format=".1f")],
            )
            .properties(height=24 * spent["candidata"].nunique() + 40),
            width="stretch",
        )
