"""Methods tab of the read-only page: hypervolume by GPU time and by unique evaluations, per method and seed (task A7).

A thin screen over ``operation.hv_analysis`` (which reads through the read model, read-only). The data layer here has
no streamlit import, so the tests exercise it directly; ``render_methods`` draws it. The same tables leave as CSV by
``cnnlab-tcc campaign curves``, with a manifest and hashes.

The value on screen is the official one: the hypervolume normalized by the reference point, in [0, 1]. The
reference point is the one frozen in the plan of the runs; when the runs have none the page asks for it and never
guesses one. Thin lines are one seed each, thick lines are the mean of the seeds of a method.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from cnnlab.tcc.operation import hv_analysis as hv

# Categorical slots 1 to 4 of the validated palette, in order (dataviz check: all gates pass in both modes on the
# adjacent pairlist; light-mode aqua and yellow are below 3:1 on the surface, which the legend, the dash patterns and
# the tables relieve). A method keeps its color wherever it appears: color follows the entity.
METHOD_COLORS = {
    "light": {"joint": "#2a78d6", "mono": "#eb6834", "random": "#1baf7a", "sequential": "#eda100"},
    "dark": {"joint": "#3987e5", "mono": "#d95926", "random": "#199e70", "sequential": "#c98500"},
}
METHOD_DASH = {"joint": [1, 0], "mono": [7, 3], "random": [2, 2], "sequential": [8, 3, 2, 3]}  # a second encoding
METHOD_NAMES = {
    "joint": "conjunto (NSGA-II)",
    "mono": "monoobjetivo (só E)",
    "random": "aleatória",
    "sequential": "sequencial",
}
AXES = {"GPU": "Tempo de GPU ativa (h)", "unique": "Avaliações únicas válidas"}
HELP = {
    "hv": "Hipervolume normalizado: cada objetivo dividido pela sua coordenada do ponto de referência, de 0 a 1. "
    "Maior é melhor. É o volume do espaço (E, L, V, S) dominado pelas candidatas e limitado pelo ponto.",
    "tempo": "Segundos de GPU ativa gastos até a avaliação, incluindo as tentativas que falharam. Pausas não contam.",
    "unicas": "Avaliações válidas que não vieram do cache da run: uma candidata repetida não conta duas vezes.",
    "ponto": "Ponto de referência do hipervolume, fixado depois do piloto e igual para todos os métodos (TCC-01 seção 5).",
    "alem": "Candidatas com E, L, V ou S igual ou maior que o ponto: não somam volume, e o protocolo manda listá-las.",
}


def parse_reference(text: str) -> tuple[float, float, float, float]:
    """``E,L,V,S`` typed on the page; a clear error for anything else."""
    try:
        values = tuple(float(part) for part in text.replace(";", ",").split(","))
    except ValueError:
        raise ValueError("escreva quatro números separados por vírgula: E, L, V, S") from None
    if len(values) != 4 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("o ponto de referência precisa de quatro números positivos: E, L, V, S")
    return values  # type: ignore[return-value]


def origin_text(origin: str) -> str:
    """Where the reference point came from, in the words of the page."""
    if origin.startswith("given"):
        return "digitado nesta página"
    return origin.replace("the frozen plan of the runs:", "do plano congelado das runs:")


def load(db_path: str, reference: Sequence[float] | None = None) -> tuple[list[hv.RunCurve], tuple[float, ...], str]:
    """The curves of every run of the database (see ``hv_analysis.read_curves``)."""
    return hv.read_curves(db_path, reference)  # type: ignore[arg-type]


def gpu_frame(curves: Sequence[hv.RunCurve], grid: Sequence[float]) -> list[dict[str, Any]]:
    """One row per run and grid time, in hours, with the method and the seed."""
    return [
        {
            "serie": r["run_id"],
            "metodo": r["method"],
            "semente": r["seed_block"],
            "x": r["gpu_s"] / 3600.0,
            "hv": r["hv"],
            "encerrada": r["run_ended"],
        }
        for r in hv.time_grid_rows(curves, grid)
    ]


def unique_frame(curves: Sequence[hv.RunCurve]) -> list[dict[str, Any]]:
    return [
        {
            "serie": r["run_id"],
            "metodo": r["method"],
            "semente": r["seed_block"],
            "x": r["unique"],
            "hv": r["hv"],
            "encerrada": False,
        }
        for r in hv.evaluation_grid_rows(curves)
    ]


def mean_frame(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean of the seeds of each method at each x, for the methods that have more than one seed there."""
    cells: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        cells[(r["metodo"], r["x"])].append(r["hv"])
    return [
        {"metodo": method, "x": x, "hv": sum(v) / len(v), "sementes": len(v)}
        for (method, x), v in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if len(v) > 1
    ]


def final_table(curves: Sequence[hv.RunCurve]) -> list[dict[str, Any]]:
    return [
        {
            "run": r["run_id"],
            "metodo": r["method"],
            "semente": r["seed_block"],
            "hv": r["hv_final"],
            "hv_no_orcamento": r["hv_at_budget"],
            "avaliacoes_unicas": r["unique_valid"],
            "frente": r["front_size"],
            "alem_do_ponto": r["beyond_reference"],
            "gpu_s": round(r["gpu_total_s"], 1),
            "orcamento_s": r["budget_gpu_s"],
            "nao_usado_s": round(r["unused_gpu_s"], 1),
            "excesso_s": round(r["excess_gpu_s"], 1),
            "etapa": r["stage"],
        }
        for r in hv.final_rows(curves)
    ]


def paired_table(curves: Sequence[hv.RunCurve]) -> list[dict[str, Any]]:
    return hv.paired_rows(curves)


# -- drawing --------------------------------------------------------------------------------------------------


def _table(st: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    st.dataframe(pd.DataFrame(list(rows)), hide_index=True)


def render_methods(st: Any, db_path: str, *, show_synthetic: bool = False) -> None:
    """Draw the tab. Errors of the read model reach the caller, which already explains them."""
    import altair as alt
    import pandas as pd

    try:
        theme = st.context.theme.type
    except Exception:  # older streamlit or no browser context
        theme = None
    colors = METHOD_COLORS.get(theme or "light", METHOD_COLORS["light"])

    st.markdown(
        "O **hipervolume** resume em um número o quanto de espaço (E, L, V, S) as candidatas de uma run dominam. "
        "Compara os métodos com o **mesmo** tempo de GPU e o mesmo ponto de referência. Maior é melhor."
    )
    typed = st.text_input(
        "Ponto de referência (E, L, V, S), se as runs não trouxerem um no plano",
        value="",
        key="methods_reference",
        help=HELP["ponto"],
        placeholder="1.0, 2.5, 12.5, 5.0",
    )
    reference = None
    if typed.strip():
        try:
            reference = parse_reference(typed)
        except ValueError as exc:
            st.error(str(exc))
            return
    try:
        curves, ref, origin = load(db_path, reference)
    except hv.AnalysisError as exc:
        st.info(f"{exc}. Informe o ponto de referência acima: ele vem do piloto e vale para todos os métodos.")
        return
    synthetic = [c for c in curves if c.synthetic]
    if synthetic and not show_synthetic:
        st.info(
            f"{len(synthetic)} run(s) têm só resultados sintéticos (teste da operação) e ficam escondidas. "
            "Marque a opção de resultados sintéticos na barra lateral para vê-las, rotuladas."
        )
        curves = [c for c in curves if not c.synthetic]
    if not curves:
        st.info("Nenhuma run com avaliações para comparar ainda.")
        return
    if synthetic and show_synthetic:
        st.warning("Resultados SINTÉTICOS (teste da operação): não são medições.")
        curves = [c for c in curves if c.synthetic]
    st.caption(
        f"Ponto de referência E={ref[0]:g}, L={ref[1]:g} ms, V={ref[2]:g} MB, S={ref[3]:g} MB, {origin_text(origin)}."
    )
    if not any(c.points for c in curves):
        st.info("As runs ainda não têm avaliação válida.")
        return

    axis = st.radio(
        "Eixo x",
        list(AXES),
        format_func=AXES.get,
        horizontal=True,
        key="methods_axis",
        help=f"{HELP['tempo']} {HELP['unicas']}",
    )
    grid = hv.time_grid(curves, hv.DEFAULT_GRID_S)
    rows = gpu_frame(curves, grid) if axis == "GPU" else unique_frame(curves)
    methods = [m for m in hv.METHOD_ORDER if any(c.method == m for c in curves)]
    domain, palette = methods, [colors[m] for m in methods]
    dashes = [METHOD_DASH[m] for m in methods]
    frame = pd.DataFrame(rows)
    means = pd.DataFrame(mean_frame(rows))
    x_title = AXES[axis]
    legend = alt.Legend(
        orient="top", title="método", symbolType="stroke", symbolStrokeWidth=3, symbolOpacity=1, direction="horizontal"
    )
    color = alt.Color("metodo:N", legend=legend, scale=alt.Scale(domain=domain, range=palette))
    dash = alt.StrokeDash("metodo:N", scale=alt.Scale(domain=domain, range=dashes), legend=None)
    seeds = (
        alt.Chart(frame)
        .mark_line(strokeWidth=1, opacity=0.45, interpolate="step-after")
        .encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("hv:Q", title="hipervolume normalizado", scale=alt.Scale(zero=True)),
            color=color,
            strokeDash=dash,
            detail="serie:N",
            tooltip=[
                "serie",
                "metodo",
                "semente",
                alt.Tooltip("x:Q", title=x_title, format=".3f"),
                alt.Tooltip("hv:Q", title="hipervolume", format=".5f"),
            ],
        )
    )
    layers = [seeds]
    if not means.empty:
        layers.append(
            alt.Chart(means)
            .mark_line(strokeWidth=3.2, interpolate="step-after")
            .encode(
                x="x:Q",
                y="hv:Q",
                color=color,
                strokeDash=dash,
                tooltip=[
                    "metodo",
                    alt.Tooltip("x:Q", title=x_title, format=".3f"),
                    alt.Tooltip("hv:Q", title="média das sementes", format=".5f"),
                    "sementes",
                ],
            )
        )
    st.altair_chart(alt.layer(*layers).properties(height=360).interactive(), width="stretch")
    st.caption(
        "Cada linha fina é uma semente; a linha grossa é a média das sementes do método. Cada método tem a sua cor "
        "e o seu traço. Uma run que termina antes do corte mantém o último valor até o fim do eixo, e o tempo que "
        "não usou aparece na tabela abaixo. No eixo de avaliações, uma run não tem valor além do que avaliou."
    )

    st.subheader("Valor final por método e semente")
    st.caption(
        "`hv` inclui a tentativa que cruzou o orçamento (ela termina e conta); `hv_no_orcamento` é o último "
        "valor confirmado dentro do orçamento. `excesso_s` é o que passou do orçamento, sem esconder."
    )
    _table(st, final_table(curves))
    paired = paired_table(curves)
    if len(paired) > 0 and len(methods) > 1:
        st.subheader("Diferença pareada por semente")
        st.caption(
            "Conjunto menos cada controle, na mesma semente: são os números de H2 e H3 antes de qualquer "
            "teste. Vazio é método sem run naquela semente, nunca zero."
        )
        _table(st, paired)

    beyond = hv.outside_rows(curves)
    st.subheader("Candidatas além do ponto de referência")
    st.caption(HELP["alem"])
    if beyond:
        st.warning(f"{len(beyond)} candidata(s) não somam volume.")
        with st.expander("Lista completa"):
            _table(st, beyond)
    else:
        st.caption("Nenhuma: todas as candidatas medidas estão dentro do ponto de referência.")
    problems = [m for c in curves for m in c.snapshot_mismatches]
    if problems:
        st.error("O hipervolume gravado pela run não bate com o calculado das avaliações: " + "; ".join(problems))
    st.caption(
        "Os mesmos números saem em CSV, com manifesto e hashes: `cnnlab-tcc campaign curves --db BANCO --dest PASTA_NOVA`."
    )
