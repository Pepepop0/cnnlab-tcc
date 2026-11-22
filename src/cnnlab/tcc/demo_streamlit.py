"""Read-only Streamlit page for the TCC v2 pilot demo.

Shows jobs, runs (with the code/system version of each run), evaluations with
the four measured objectives E, L, V, S, what each candidate is (its recorded
genome, with the precision of every block), a scatter of the valid
measurements, and the generations, checkpoints and front snapshots of a run. Everything comes from
`cnnlab.tcc.storage.readmodel` on a `mode=ro` + `query_only` SQLite
connection: this page cannot write, holds no lease, migrates nothing and
takes no SQL from the operator. The database path is always given explicitly
(text box or `--db`).

A database of schema v1 (a run written without the operation queue) is shown
as it is: no jobs, the runs and their results.

Results of a synthetic (fake) evaluator are never listed with measurements:
they stay hidden unless the operator ticks the box, and then appear in their
own section, labeled as synthetic in the heading and in every column. What
the evaluator itself declared about real measurements (preliminary, not
confirmatory) is shown next to them.

Run::

    streamlit run src/cnnlab/tcc/demo_streamlit.py -- --db D:\\tcc-demo\\db\\tcc.sqlite3
    python -m cnnlab.tcc.demo_streamlit --db D:\\tcc-demo\\db\\tcc.sqlite3   # same, launches streamlit
    python -m cnnlab.tcc.demo_streamlit --db ... --server.port 8502 --server.headless true

Options other than `--db` are handed to `streamlit run` unchanged; the second
and third forms listen on 127.0.0.1 unless `--server.address` is given (plain
`streamlit run` listens on every interface). Streamlit is imported only when
the page runs, so this module (and its data layer) imports without it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cnnlab  # noqa: F401
except ModuleNotFoundError:  # `streamlit run <this file>` from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cnnlab.tcc.demo_analysis import render_analysis
from cnnlab.tcc.demo_glossary import render_glossary
from cnnlab.tcc.demo_methods import render_methods
from cnnlab.tcc.storage.readmodel import (
    READ_MODEL_VERSION,
    ReadModelError,
    generation_summary,
    open_read_model,
    per_block,
    version_text,
)

PAGE_TITLE = "TCC v2 - demo piloto (somente leitura)"
STREAMLIT_INSTALL_HINT = 'pip install "streamlit>=1.40"'
LOCAL_ADDRESS = "127.0.0.1"
SYNTHETIC_LABEL = "SINTETICO (teste)"
REAL_LABEL = "real"
HIDDEN = "oculto"
NOT_RECORDED = "(nao registrado)"
NOT_COMPUTED = "nao calculado"
TABLE_ROW_PX, TABLE_MAX_PX = 35, 1400
# Decimals shown on screen; the database keeps the full value.
DISPLAY_DECIMALS = {"e": 4, "l_ms": 4, "v_mb": 3, "s_mb": 3, "gpu_seconds": 1}

# What went wrong and what to do next, by `ReadModelError.code`.
ERROR_TEXT: dict[str, tuple[str, str]] = {
    "empty_path": (
        "Informe o caminho do banco",
        "Digite o caminho completo do arquivo `tcc.sqlite3` na barra lateral, ou abra a página com `-- --db CAMINHO`.",
    ),
    "not_found": (
        "Arquivo não encontrado",
        "Confira o caminho. O banco nasce com `python -m cnnlab.tcc.operation.demo prepare`; "
        "esta página nunca cria arquivos.",
    ),
    "not_a_file": (
        "O caminho é uma pasta, não um arquivo",
        "Aponte para o arquivo do banco (por exemplo `...\\db\\tcc.sqlite3`).",
    ),
    "not_sqlite": (
        "O arquivo não abre como banco SQLite",
        "Confira se é o `tcc.sqlite3` da demo e se os arquivos `-wal`/`-shm` ao lado dele não foram apagados.",
    ),
    "not_tcc": (
        "O arquivo é SQLite, mas não é um banco do TCC v2",
        "Use o banco criado pelo comando `prepare` da operação do TCC v2 (não o JSONL/SQLite da POC).",
    ),
    "schema_too_old": (
        "Banco sem esquema do TCC v2",
        "Nenhuma migração foi aplicada neste arquivo. Use um banco criado pelo `RunStore` ou pelo comando `prepare`.",
    ),
    "schema_too_new": (
        "Banco mais novo que este código",
        "Atualize o checkout para a versão que criou o banco; ler com código antigo poderia interpretar errado.",
    ),
}

JOB_COLUMNS = (
    "job",
    "estado",
    "run",
    "metodo",
    "bloco",
    "avaliador",
    "gpu_ativa_s",
    "orcamento_gpu_s",
    "validas",
    "invalidas",
    "interrompidas",
    "em_curso",
    "lease",
    "heartbeats",
    "claims",
    "cancelamento",
    "detalhe",
)
RUN_COLUMNS = (
    "run",
    "estagio",
    "metodo",
    "bloco",
    "orcamento_gpu_s",
    "versao_do_sistema",
    "git_sha",
    "codigo_sujo",
    "rotulo_do_codigo",
    "fiacao",
    "avaliador",
    "runner",
    "plano",
    "plano_confirmatorio",
    "dataset",
    "ambiente",
    "criada_em",
)
EVALUATION_COLUMNS = (
    "tentativa",
    "candidata",
    "geracao",
    "status",
    "E_erro_top1",
    "L_ms",
    "V_MB",
    "S_MB",
    "gpu_s",
    "precisao_por_bloco",
    "cache",
    "falha",
    "observacao",
    "confirmada_em",
)
SYNTHETIC_EVALUATION_COLUMNS = tuple(
    {
        "E_erro_top1": "E_SINTETICO",
        "L_ms": "L_ms_SINTETICO",
        "V_MB": "V_MB_SINTETICO",
        "S_MB": "S_MB_SINTETICO",
        "gpu_s": "gpu_s_SINTETICO",
    }.get(c, c)
    for c in EVALUATION_COLUMNS
)
GENERATION_COLUMNS = (
    "geracao",
    "candidatas",
    "tentativas",
    "validas",
    "invalidas",
    "interrompidas",
    "em_curso",
    "cache",
    "gpu_s",
    "menor_E",
    "menor_L_ms",
    "menor_V_MB",
    "menor_S_MB",
)
CHECKPOINT_COLUMNS = (
    "checkpoint",
    "geracao",
    "avaliacoes_unicas",
    "gpu_ativa_s",
    "populacao",
    "arquivo_pareto",
    "pendentes",
    "criado_em",
)
GENOME_COLUMNS = (
    "candidata",
    "geracao",
    "origem",
    "blocos",
    "precisao_por_bloco",
    "precisao_stem",
    "precisao_cabeca",
    "canais_por_bloco",
    "op_por_bloco",
    "kernel_por_bloco",
    "convs_por_bloco",
    "residual_por_bloco",
    "downsample_por_bloco",
    "poda_pct_por_bloco",
    "stem_canais",
    "ativacao",
    "cabeca",
    "reparos",
)
# Scatter: x is chosen by the operator, y is always E. Colors validated with the dataviz
# palette checker on both the light and the dark surface; shape repeats the identity.
SCATTER_AXES = {"L (ms)": "L_ms", "V (MB)": "V_MB", "S (MB)": "S_MB"}
IN_FRONT, OFF_FRONT = "na frente mais recente", "fora da frente"
FRONT_DOMAIN = (OFF_FRONT, IN_FRONT)
FRONT_COLORS = ("#3987e5", "#d95926")
FRONT_SHAPES = ("circle", "diamond")
MARK_SIZE, MAX_DIRECT_LABELS = 200, 4
SURFACES = {"light": ("#ffffff", "#0b0b0b"), "dark": ("#0e1117", "#ffffff")}  # surface, text
FRONT_COLUMNS = ("frente", "gpu_ativa_s", "unicas_validas", "hipervolume", "membros", "candidatas", "registrada_em")

# Help shown by the (i) of each column header, so nobody has to memorize what E, L, V, S are.
_E = "Erro de classificação top-1 do engine nas 5.000 imagens de validação, de 0 a 1. Menor é melhor: 0,78 = 22% de acerto."
_L = "Latência: mediana do tempo de GPU de uma inferência com lote 1, em milissegundos. Menor é melhor."
_V = "Memória de vídeo: pico alocado na GPU durante a inferência, acima do que o processo já usava, em MB. Menor é melhor."
_S = "Armazenamento: tamanho do arquivo .plan do engine TensorRT, em MB. Menor é melhor."
_BEST = " Considera só as candidatas avaliadas nesta geração; as melhores das gerações anteriores continuam na população e na frente, mas não são reavaliadas."
COLUMN_HELP: dict[str, str] = {
    "E_erro_top1": _E,
    "L_ms": _L,
    "V_MB": _V,
    "S_MB": _S,
    "E_SINTETICO": "Valor fabricado pela fiação de teste. Não é medição.",
    "L_ms_SINTETICO": "Valor fabricado pela fiação de teste. Não é medição.",
    "V_MB_SINTETICO": "Valor fabricado pela fiação de teste. Não é medição.",
    "S_MB_SINTETICO": "Valor fabricado pela fiação de teste. Não é medição.",
    "menor_E": "Menor erro entre as candidatas desta geração." + _BEST,
    "menor_L_ms": "Menor latência entre as candidatas desta geração." + _BEST,
    "menor_V_MB": "Menor memória entre as candidatas desta geração." + _BEST,
    "menor_S_MB": "Menor tamanho entre as candidatas desta geração." + _BEST,
    "gpu_s": "Tempo de GPU ativa gasto: treino, exportação, compilação e medição, em segundos.",
    "gpu_ativa_s": "Tempo de GPU ativa acumulado, em segundos. Conta para o orçamento da run.",
    "orcamento_gpu_s": "Limite de tempo de GPU ativa da run, em segundos. A avaliação que cruza o limite termina e conta.",
    "tentativa": "Uma avaliação de uma candidata. O sufixo .a1, .a2 numera tentativas repetidas após falha de infraestrutura.",
    "candidata": "Genoma registrado na run: <run>.g<geração>.c<índice>.",
    "geracao": "Geração da busca em que a candidata foi criada. 0 é a população inicial.",
    "status": "valid: medida. invalid: a rede falhou (sem valor inventado). interrupted: a infraestrutura falhou; é avaliada de novo.",
    "precisao_por_bloco": "Precisão numérica de cada bloco ativo, na ordem: fp32 (32 bits) ou fp16 (16 bits, metade da memória).",
    "precisao_stem": "Precisão da primeira convolução. Fixa em fp32 nos recortes atuais.",
    "precisao_cabeca": "Precisão do classificador final. Fixa em fp32 nos recortes atuais.",
    "cache": "Sim quando o mesmo genoma já tinha sido medido nesta run e o resultado foi reaproveitado sem gastar GPU.",
    "observacao": "O que o avaliador declarou (preliminar, não confirmatória) e se a candidata está na frente mais recente.",
    "validas": "Tentativas com os quatro objetivos medidos.",
    "invalidas": "Tentativas em que a rede falhou: não treinou, não compilou ou não executou como pedido.",
    "interrompidas": "Tentativas em que a infraestrutura falhou. Nada foi observado sobre a candidata; ela é avaliada de novo.",
    "em_curso": "Tentativas em execução agora.",
    "lease": "Posse temporária do job por um worker. 'vivo' = o worker está renovando; se ele morrer, o prazo vence e outro assume.",
    "heartbeats": "Quantas vezes o worker renovou o lease. Subindo = worker vivo.",
    "claims": "Quantas vezes um worker assumiu este job. Mais de 1 = houve retomada.",
    "codigo_sujo": "Sim quando havia arquivo versionado modificado sem commit ao preparar a run.",
    "git_sha": "Commit exato do código que produziu a run.",
    "blocos": "Número de blocos ativos da rede (2 a 6).",
    "canais_por_bloco": "Canais de saída de cada bloco.",
    "op_por_bloco": "Tipo de unidade de cada bloco: standard, grouped_g4, dw_separable, mbconv_e3, mbconv_e6.",
    "kernel_por_bloco": "Tamanho do filtro espacial de cada bloco (3, 5 ou 7).",
    "convs_por_bloco": "Quantas unidades cada bloco repete (1 a 3).",
    "residual_por_bloco": "Se cada bloco tem atalho somando a entrada à saída.",
    "downsample_por_bloco": "Como cada bloco reduz a resolução pela metade, ou none.",
    "poda_pct_por_bloco": "Percentual de canais removidos após o treino. Fixo em 0 nos recortes atuais.",
    "reparos": "Quantos reparos determinísticos o genoma precisou para virar uma rede construível.",
    "arquivo_pareto": "Candidatas não dominadas entre todas as já avaliadas na run.",
    "pendentes": "Candidatas registradas e ainda não avaliadas quando o checkpoint foi salvo.",
    "hipervolume": "Volume do espaço de objetivos dominado pela frente. Só é calculado depois do piloto, quando o ponto de referência for fixado.",
    "membros": "Quantas candidatas estão nesta frente.",
}


# -- data layer (no streamlit) ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One consistent read of the queue and the runs."""

    db_path: str
    read_model: str
    schema_version: int
    has_job_queue: bool
    jobs: tuple[Mapping[str, Any], ...]
    runs: tuple[Mapping[str, Any], ...]
    progress: Mapping[str, Mapping[str, Any]]

    def is_synthetic_run(self, run_id: str) -> bool:
        """A run is synthetic when its job says so or any of its results is labeled synthetic."""
        by_job = any(j["run_id"] == run_id and j["options"].get("synthetic_evaluator") for j in self.jobs)
        return bool(by_job or self.progress.get(run_id, {}).get("synthetic_results", 0))


@dataclass(frozen=True, slots=True)
class RunView:
    run: Mapping[str, Any]
    progress: Mapping[str, Any]
    synthetic_run: bool
    has_job_queue: bool
    real_evaluations: tuple[Mapping[str, Any], ...]
    synthetic_evaluations: tuple[Mapping[str, Any], ...]
    checkpoints: tuple[Mapping[str, Any], ...]
    fronts: tuple[Mapping[str, Any], ...]
    genomes: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    job_events: tuple[Mapping[str, Any], ...]

    @property
    def latest_front_ids(self) -> frozenset[str]:
        return frozenset(self.fronts[-1]["candidate_ids"]) if self.fronts else frozenset()


def load_snapshot(db_path: str | Path | None) -> Snapshot:
    """Jobs, runs and per-run progress; raises `ReadModelError` with a code for the page."""
    with open_read_model(db_path) as rm:
        runs = tuple(rm.runs())
        return Snapshot(
            db_path=str(db_path),
            read_model=rm.version,
            schema_version=rm.schema_version(),
            has_job_queue=rm.has_job_queue,
            jobs=tuple(rm.jobs()),
            runs=runs,
            progress={r["run_id"]: rm.run_progress(r["run_id"]) for r in runs},
        )


def load_run(db_path: str | Path | None, run_id: str) -> RunView:
    """Everything the page shows about one run, real and synthetic results kept apart."""
    with open_read_model(db_path) as rm:
        run = next((r for r in rm.runs() if r["run_id"] == run_id), None)
        if run is None:
            raise ReadModelError("run_not_found", f"run {run_id!r} is not in this database")
        jobs = [j for j in rm.jobs() if j["run_id"] == run_id]
        evaluations = rm.evaluations(run_id)
        progress = rm.run_progress(run_id)
        events: list[Mapping[str, Any]] = []
        for job in jobs:
            events.extend({"job": job["job_id"], **e} for e in rm.job_events(job["job_id"]))
        return RunView(
            run=run,
            progress=progress,
            synthetic_run=bool(
                any(j["options"].get("synthetic_evaluator") for j in jobs) or progress["synthetic_results"]
            ),
            has_job_queue=rm.has_job_queue,
            real_evaluations=tuple(r for r in evaluations if not r["synthetic"]),
            synthetic_evaluations=tuple(r for r in evaluations if r["synthetic"]),
            checkpoints=tuple(rm.checkpoints(run_id)),
            fronts=tuple(rm.fronts(run_id)),
            genomes=tuple(rm.genomes(run_id)),
            environment=rm.environment(run["environment_id"]),
            job_events=tuple(events),
        )


def explain_error(exc: ReadModelError) -> tuple[str, str]:
    """Title and next step for the operator; unknown codes fall back to the raw message."""
    title, hint = ERROR_TEXT.get(exc.code, ("Não foi possível ler o banco", "Confira o caminho e tente de novo."))
    return title, f"{hint}\n\nDetalhe: `{exc}`"


def _lease_text(job: Mapping[str, Any]) -> str:
    if job["state"] != "running":
        return ""
    live = "vivo" if job["lease_live"] else "VENCIDO (worker parado?)"
    return f"{job['lease_owner']} - {live} ate {job['lease_expires_at']}"


def job_rows(snapshot: Snapshot, *, show_synthetic: bool) -> list[dict[str, Any]]:
    """Queue table. GPU time and counts of a synthetic run are fake too: hidden unless asked."""
    runs = {r["run_id"]: r for r in snapshot.runs}
    rows = []
    for job in snapshot.jobs:
        run = runs[job["run_id"]]
        progress = snapshot.progress[job["run_id"]]
        synthetic = snapshot.is_synthetic_run(job["run_id"])
        visible = show_synthetic or not synthetic
        attempts = progress["attempts"]
        rows.append(
            {
                "job": job["job_id"],
                "estado": job["state"],
                "run": job["run_id"],
                "metodo": run["method"],
                "bloco": run["seed_block"],
                "avaliador": SYNTHETIC_LABEL if synthetic else REAL_LABEL,
                # Text in these columns: one column never mixes numbers and "oculto".
                "gpu_ativa_s": f"{progress['active_gpu_s']:.1f}" if visible else HIDDEN,
                "orcamento_gpu_s": run["budget_gpu_s"],
                "validas": str(attempts.get("valid", 0)) if visible else HIDDEN,
                "invalidas": str(attempts.get("invalid", 0)) if visible else HIDDEN,
                "interrompidas": str(attempts.get("interrupted", 0)) if visible else HIDDEN,
                "em_curso": str(attempts.get("running", 0)) if visible else HIDDEN,
                "lease": _lease_text(job),
                "heartbeats": job["heartbeat_count"],
                "claims": job["claim_count"],
                "cancelamento": job["cancel_reason"] if job["cancel_requested"] else "",
                "detalhe": job["final_detail"] or "",
            }
        )
    return rows


def run_rows(snapshot: Snapshot) -> list[dict[str, Any]]:
    """Runs with what the database records about the code that created each one."""
    return [
        {
            "run": r["run_id"],
            "estagio": r["stage"],
            "metodo": r["method"],
            "bloco": r["seed_block"],
            "orcamento_gpu_s": r["budget_gpu_s"],
            "versao_do_sistema": version_text(r, missing=NOT_RECORDED),
            "git_sha": r["git_sha"] or NOT_RECORDED,
            "codigo_sujo": {True: "sim", False: "nao"}.get(r["code_dirty"], NOT_RECORDED),
            "rotulo_do_codigo": r["code_label"] or "",
            "fiacao": r["wiring_label"] or "",
            "avaliador": r["evaluator_label"] or "",
            "runner": r["runner_label"] or "",
            "plano": r["plan_id"],
            "plano_confirmatorio": {True: "sim", False: "nao"}.get(r["plan_confirmatory"], NOT_RECORDED),
            "dataset": r["dataset_id"],
            "ambiente": r["environment_id"],
            "criada_em": r["created_at"],
        }
        for r in snapshot.runs
    ]


def _shown(row: Mapping[str, Any], key: str) -> float | None:
    return None if row[key] is None else round(float(row[key]), DISPLAY_DECIMALS[key])


def evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    synthetic: bool = False,
    front_ids: frozenset[str] = frozenset(),
    genomes: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Attempt table. Unmeasured objectives stay empty (never zero); synthetic columns say so.

    The precision per block comes from the recorded genome of the candidate,
    so a measurement is read next to what was measured.
    """
    columns = SYNTHETIC_EVALUATION_COLUMNS if synthetic else EVALUATION_COLUMNS
    by_candidate = {g["candidate_id"]: g for g in genomes}
    out = []
    for r in rows:
        if bool(r["synthetic"]) != synthetic:
            raise ValueError("real and synthetic evaluations are never listed in the same table")
        failure = f"{r['failure_stage']}/{r['failure_code']}" if r["failure_code"] else ""
        notes = [
            text
            for text, on in (
                ("preliminar", r.get("preliminary")),
                ("nao confirmatoria", r.get("not_confirmatory")),
                ("na frente mais recente", r["candidate_id"] in front_ids),
                ("recuperada apos queda", r.get("recovered")),
            )
            if on
        ]
        values = (
            r["attempt_id"],
            r["candidate_id"],
            r["generation"],
            r["status"],
            _shown(r, "e"),
            _shown(r, "l_ms"),
            _shown(r, "v_mb"),
            _shown(r, "s_mb"),
            _shown(r, "gpu_seconds"),
            per_block(by_candidate.get(r["candidate_id"], {}), "precision", missing=NOT_RECORDED),
            "sim" if r["cache_hit"] else "",
            failure,
            "; ".join(notes),
            r["committed_at"] or "",
        )
        out.append(dict(zip(columns, values, strict=True)))
    return out


def generation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-generation summary of the given evaluations (the four minima are independent)."""
    out = []
    for g in generation_summary(rows):
        values = (
            g["generation"],
            g["candidates"],
            g["attempts"],
            g["valid"],
            g["invalid"],
            g["interrupted"],
            g["running"],
            g["cache_hits"],
            round(g["gpu_seconds"], 1),
            *(
                None if g[f"min_{k}"] is None else round(g[f"min_{k}"], DISPLAY_DECIMALS[k])
                for k in ("e", "l_ms", "v_mb", "s_mb")
            ),
        )
        out.append(dict(zip(GENERATION_COLUMNS, values, strict=True)))
    return out


def _text(value: Any) -> str:
    return NOT_RECORDED if value is None else str(value)


def genome_rows(genomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What each candidate is, from its recorded genome; text columns, nothing completed."""
    out = []
    for g in genomes:
        values = (
            g["candidate_id"],
            _text(g["generation"]),
            _text(g["origin"]),
            _text(g["n_blocks"]),
            per_block(g, "precision", missing=NOT_RECORDED),
            _text(g["stem_precision"]),
            _text(g["head_precision"]),
            per_block(g, "channels", missing=NOT_RECORDED),
            per_block(g, "op", missing=NOT_RECORDED),
            per_block(g, "kernel", missing=NOT_RECORDED),
            per_block(g, "n_convs", missing=NOT_RECORDED),
            per_block(g, "residual", missing=NOT_RECORDED),
            per_block(g, "downsample", missing=NOT_RECORDED),
            per_block(g, "prune_pct", missing=NOT_RECORDED),
            _text(g["stem_channels"]),
            _text(g["activation"]),
            _text(g["head"]),
            _text(g["repairs"]),
        )
        out.append(dict(zip(GENOME_COLUMNS, values, strict=True)))
    return out


def scatter_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    front_ids: frozenset[str] = frozenset(),
    genomes: Sequence[Mapping[str, Any]] = (),
    run_id: str = "",
) -> list[dict[str, Any]]:
    """One point per valid attempt that has the four objectives; full values, no rounding."""
    by_candidate = {g["candidate_id"]: g for g in genomes}
    prefix = f"{run_id}."
    points = []
    for r in rows:
        if r["status"] != "valid" or any(r[k] is None for k in ("e", "l_ms", "v_mb", "s_mb")):
            continue
        cid = r["candidate_id"]
        genome = by_candidate.get(cid, {})
        points.append(
            {
                "candidata": cid[len(prefix) :] if run_id and cid.startswith(prefix) else cid,
                "geracao": r["generation"],
                "E": float(r["e"]),
                "L_ms": float(r["l_ms"]),
                "V_MB": float(r["v_mb"]),
                "S_MB": float(r["s_mb"]),
                "frente": IN_FRONT if cid in front_ids else OFF_FRONT,
                "precisao_por_bloco": per_block(genome, "precision", missing=NOT_RECORDED),
            }
        )
    return points


def checkpoint_rows(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(
            zip(
                CHECKPOINT_COLUMNS,
                (
                    c["checkpoint_id"],
                    c["generation"],
                    c["unique_evaluations"],
                    round(c["active_gpu_seconds"], 1),
                    c["population"],
                    c["pareto_archive"],
                    c["pending"],
                    c["created_at"],
                ),
                strict=True,
            )
        )
        for c in checkpoints
    ]


def front_rows(fronts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Front snapshots; a hypervolume that was not computed says so instead of showing a number."""
    return [
        dict(
            zip(
                FRONT_COLUMNS,
                (
                    f["snapshot_id"],
                    round(f["active_gpu_seconds"], 1),
                    f["unique_valid_evaluations"],
                    NOT_COMPUTED if f["hypervolume"] is None else f"{f['hypervolume']:.6g}",
                    len(f["candidate_ids"]),
                    ", ".join(f["candidate_ids"]),
                    f["created_at"],
                ),
                strict=True,
            )
        )
        for f in fronts
    ]


# -- page ---------------------------------------------------------------------------------


def _table(st: Any, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    import pandas as pd  # streamlit depends on pandas

    # Tall enough to show every row of a demo-sized table without scrolling inside it.
    height = min(TABLE_ROW_PX * (len(rows) + 1) + 3, TABLE_MAX_PX)
    config = {name: st.column_config.Column(help=COLUMN_HELP[name]) for name in columns if name in COLUMN_HELP}
    st.dataframe(pd.DataFrame(list(rows), columns=list(columns)), hide_index=True, height=height, column_config=config)


def _theme(st: Any) -> str:
    try:
        kind = st.context.theme.type
    except Exception:  # older streamlit, or no browser context
        kind = None
    return kind if kind in SURFACES else "light"


def _render_scatter(st: Any, points: Sequence[Mapping[str, Any]], label: str) -> None:
    """E against L, V or S for the valid attempts; front membership by color and by shape."""
    import altair as alt
    import pandas as pd

    st.markdown(f"**Dispersão das medições válidas{label}**")
    axis = st.radio("Eixo X", list(SCATTER_AXES), horizontal=True, key="scatter_x")
    field = SCATTER_AXES[axis]
    surface, ink = SURFACES[_theme(st)]
    frame = pd.DataFrame(list(points))
    identity = {"scale": alt.Scale(domain=list(FRONT_DOMAIN), range=list(FRONT_COLORS))}
    legend = alt.Legend(title="Frente registrada mais recente", orient="bottom", symbolSize=MARK_SIZE)
    base = alt.Chart(frame).encode(
        x=alt.X(f"{field}:Q", title=axis, scale=alt.Scale(zero=False)),
        y=alt.Y("E:Q", title="E (fração de erro top-1)", scale=alt.Scale(zero=False)),
    )
    marks = base.mark_point(filled=True, size=MARK_SIZE, opacity=1, stroke=surface, strokeWidth=2).encode(
        color=alt.Color("frente:N", legend=legend, **identity),
        shape=alt.Shape(
            "frente:N", legend=legend, scale=alt.Scale(domain=list(FRONT_DOMAIN), range=list(FRONT_SHAPES))
        ),
        tooltip=[
            alt.Tooltip("candidata:N"),
            alt.Tooltip("geracao:Q", title="geração"),
            alt.Tooltip("E:Q", format=".4f"),
            alt.Tooltip("L_ms:Q", format=".4f", title="L (ms)"),
            alt.Tooltip("V_MB:Q", format=".3f", title="V (MB)"),
            alt.Tooltip("S_MB:Q", format=".3f", title="S (MB)"),
            alt.Tooltip("precisao_por_bloco:N", title="precisão por bloco"),
            alt.Tooltip("frente:N"),
        ],
    )
    names = (
        base.transform_filter(alt.datum.frente == IN_FRONT)
        .mark_text(align="left", dx=9, dy=-9, color=ink)
        .encode(text="candidata:N")
    )
    in_front = sum(1 for p in points if p["frente"] == IN_FRONT)
    labeled = 0 < in_front <= MAX_DIRECT_LABELS  # more labels than that collide; the tooltip names every point
    st.altair_chart(((marks + names) if labeled else marks).properties(height=380))
    st.caption(
        "Menor é melhor nos dois eixos. "
        + ("Os pontos da frente levam rótulo. " if labeled else "")
        + "Passe o mouse sobre um ponto para ver candidata, valores e precisão por bloco. "
        "Os mesmos valores estão na tabela de medições."
    )


def _render_genomes(st: Any, view: RunView, label: str) -> None:
    st.markdown(f"**Genoma das candidatas{label}**")
    if not view.genomes:
        st.caption("Nenhuma candidata registrada nesta run.")
        return
    st.caption(
        'O que cada candidata é, lido do genoma expresso gravado no banco. Uma coluna "por bloco" lista os '
        f"blocos ativos na ordem. `{NOT_RECORDED}` = o banco não registra esse campo."
    )
    _table(st, genome_rows(view.genomes), GENOME_COLUMNS)


def _render_search_state(st: Any, view: RunView, label: str) -> None:
    rows = view.synthetic_evaluations if view.synthetic_run else view.real_evaluations
    st.markdown(f"**Gerações{label}**")
    if rows:
        st.caption(
            "Os quatro mínimos são independentes (em geral vêm de candidatas diferentes): resumem a geração, "
            "não são um ponto da frente. Cada linha considera só as candidatas novas daquela geração: as melhores "
            "das gerações anteriores sobrevivem sem ser reavaliadas, então um mínimo pode subir de uma geração "
            "para outra sem que a busca tenha piorado. O progresso aparece na frente."
        )
        _table(st, generation_rows(rows), GENERATION_COLUMNS)
    else:
        st.caption("Nenhuma tentativa registrada.")
    st.markdown(f"**Checkpoints{label}**")
    if view.checkpoints:
        _table(st, checkpoint_rows(view.checkpoints), CHECKPOINT_COLUMNS)
    else:
        st.caption("Nenhum checkpoint registrado nesta run.")
    st.markdown(f"**Frentes registradas{label}**")
    if view.fronts:
        st.caption("Cada linha é um retrato do arquivo de não dominados no momento indicado.")
        _table(st, front_rows(view.fronts), FRONT_COLUMNS)
    else:
        st.caption("Nenhuma frente registrada nesta run.")


def _render_run(st: Any, db_path: str, snapshot: Snapshot, show_synthetic: bool) -> None:
    st.subheader("Avaliações da run")
    run_ids = [r["run_id"] for r in snapshot.runs]
    run_id = st.selectbox("Run", run_ids, index=len(run_ids) - 1, key="run_id")
    view = load_run(db_path, run_id)
    run = view.run

    st.markdown(f"**Versão do sistema:** `{version_text(run, missing=NOT_RECORDED)}`")
    if not run["system_version"]:
        st.caption(
            "Este banco não registra a versão legível do pacote nem se o código estava sujo; "
            "o que aparece acima é o que o manifesto de ambiente registra."
        )
    st.caption(
        f"estágio {run['stage']} - método {run['method']} - bloco {run['seed_block']} - "
        f"plano {run['plan_id']} - dataset {run['dataset_id']}"
    )
    if run["stage"] == "pilot":
        st.caption("Run piloto exploratória: não faz parte dos blocos confirmatórios nem das hipóteses H1-H4.")

    if view.synthetic_run:
        st.warning(
            f"Run com avaliador SINTÉTICO (teste): {len(view.synthetic_evaluations)} resultado(s) falsos. "
            "Não são medições de E, L, V, S e ficam fora da tabela de medições."
        )
    visible = show_synthetic or not view.synthetic_run
    label = " (SINTÉTICO)" if view.synthetic_run else ""
    if not visible:
        st.info(
            "Os números desta run (objetivos, tempo de GPU e contagens) estão ocultos por serem sintéticos. "
            'Marque "Mostrar resultados sintéticos (teste)" na barra lateral para vê-los, rotulados.'
        )
    else:
        attempts = view.progress["attempts"]
        cols = st.columns(6)
        cols[0].metric(f"GPU ativa (s){label}", f"{view.progress['active_gpu_s']:.1f}")
        cols[1].metric("Orçamento GPU (s)", f"{view.progress['budget_gpu_s']:.0f}")
        cols[2].metric(f"Válidas{label}", attempts.get("valid", 0))
        cols[3].metric(f"Inválidas{label}", attempts.get("invalid", 0))
        cols[4].metric(f"Interrompidas{label}", attempts.get("interrupted", 0))
        cols[5].metric(f"Únicas válidas{label}", view.progress["unique_valid_evaluations"])

    st.markdown("**Medições (E, L, V, S)**")
    st.markdown(
        "Os quatro objetivos são **minimizados**: menor é melhor em todos.  \n"
        "**E** erro de classificação (0,78 = 22% de acerto) · **L** latência por inferência, em ms · "
        "**V** pico de memória de vídeo, em MB · **S** tamanho do arquivo do engine, em MB.  \n"
        "Passe o mouse no nome de uma coluna para ver o que ela significa."
    )
    if view.real_evaluations:
        st.caption(
            "E = fração de erro top-1; L = ms; V e S = MB decimais. `None` = não medido (nunca zero). "
            "Valores arredondados só na tela; o banco guarda o valor completo."
        )
        if any(r["preliminary"] or r["not_confirmatory"] for r in view.real_evaluations):
            st.info(
                "O próprio avaliador marcou estas medições como **preliminares** e **não confirmatórias** "
                "(`quality.preliminary`, `quality.not_confirmatory`). São medições reais de uma run piloto."
            )
        _table(
            st,
            evaluation_rows(view.real_evaluations, front_ids=view.latest_front_ids, genomes=view.genomes),
            EVALUATION_COLUMNS,
        )
    elif view.synthetic_run:
        st.info("Nenhuma medição real nesta run: ela só tem resultados sintéticos de teste.")
    else:
        st.info("Nenhuma tentativa registrada ainda. Inicie o worker (`start`) e atualize a página.")

    if view.synthetic_evaluations and show_synthetic:
        st.markdown("**Resultados SINTÉTICOS (teste) - NÃO são medições**")
        _table(
            st,
            evaluation_rows(
                view.synthetic_evaluations, synthetic=True, front_ids=view.latest_front_ids, genomes=view.genomes
            ),
            SYNTHETIC_EVALUATION_COLUMNS,
        )

    if visible:
        shown = view.synthetic_evaluations if view.synthetic_run else view.real_evaluations
        points = scatter_points(shown, front_ids=view.latest_front_ids, genomes=view.genomes, run_id=run["run_id"])
        if points:
            _render_scatter(st, points, label)
    _render_genomes(st, view, label)
    if visible:
        _render_search_state(st, view, label)

    with st.expander("Manifesto de ambiente e versão do código"):
        st.json(dict(view.environment))
    with st.expander("Eventos do job"):
        if view.job_events:
            _table(
                st,
                [
                    {
                        "job": e["job"],
                        "quando": e["recorded_at"],
                        "evento": e["kind"],
                        "de": e["from_state"] or "",
                        "para": e["to_state"],
                        "worker": e["worker_id"] or "",
                    }
                    for e in view.job_events
                ],
                ("job", "quando", "evento", "de", "para", "worker"),
            )
        elif not view.has_job_queue:
            st.caption("Banco sem fila de jobs (esquema v1): não há eventos de job, lease nem heartbeat.")
        else:
            st.caption("Esta run ainda não tem job na fila.")


def _render_content(st: Any, db_path: str, show_synthetic: bool) -> None:
    try:
        snapshot = load_snapshot(db_path)
    except ReadModelError as exc:
        title, hint = explain_error(exc)
        if exc.code == "empty_path":
            st.info(f"**{title}.** {hint}")
        else:
            st.error(f"**{title}.** {hint}")
        return
    st.caption(f"{snapshot.read_model} - esquema v{snapshot.schema_version} - `{snapshot.db_path}`")
    operation, analysis, methods, glossary = st.tabs(["Operação", "Análise", "Métodos", "Glossário"])
    with operation:
        _render_operation(st, db_path, snapshot, show_synthetic)
    with analysis:
        if not snapshot.runs:
            st.info("Nenhuma run registrada neste banco ainda.")
        else:
            try:
                render_analysis(st, db_path, [r["run_id"] for r in snapshot.runs])
            except ReadModelError as exc:
                title, hint = explain_error(exc)
                st.error(f"**{title}.** {hint}")
    with methods:
        if not snapshot.runs:
            st.info("Nenhuma run registrada neste banco ainda.")
        else:
            try:
                render_methods(st, db_path, show_synthetic=show_synthetic)
            except ReadModelError as exc:
                title, hint = explain_error(exc)
                st.error(f"**{title}.** {hint}")
    with glossary:
        render_glossary(st)


def _render_operation(st: Any, db_path: str, snapshot: Snapshot, show_synthetic: bool) -> None:
    st.subheader("Jobs")
    if snapshot.jobs:
        _table(st, job_rows(snapshot, show_synthetic=show_synthetic), JOB_COLUMNS)
    elif not snapshot.has_job_queue:
        st.info(
            "Este banco não tem fila de jobs (esquema v1): a run foi gravada direto pelo `RunStore`, "
            "sem job, lease nem heartbeat. As runs e os resultados aparecem abaixo. Esta página não migra o banco."
        )
    else:
        st.info(
            "Nenhum job na fila. Prepare um com "
            "`python -m cnnlab.tcc.operation.demo prepare --db ... --artifacts ... --spec ...` ou pela TUI."
        )

    st.subheader("Runs e versão")
    if not snapshot.runs:
        st.info("Nenhuma run registrada neste banco ainda.")
        return
    _table(st, run_rows(snapshot), RUN_COLUMNS)
    try:
        _render_run(st, db_path, snapshot, show_synthetic)
    except ReadModelError as exc:
        title, hint = explain_error(exc)
        st.error(f"**{title}.** {hint}")


def render(st: Any, default_db: str = "") -> None:
    """Draw the page with the given streamlit module (injected so tests can drive it)."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title("TCC v2 - demo piloto")
    st.caption(
        f"Somente leitura ({READ_MODEL_VERSION}, SQLite query_only). Esta página não altera o banco, "
        "não segura lease e não aceita SQL."
    )
    with st.sidebar:
        db_path = st.text_input("Caminho do banco SQLite (tcc.sqlite3)", value=default_db, key="db_path")
        show_synthetic = st.checkbox("Mostrar resultados sintéticos (teste)", value=False, key="show_synthetic")
        every = st.number_input(
            "Atualizar sozinho a cada (s); 0 = manual", min_value=0, max_value=3600, value=0, step=5, key="refresh_s"
        )
        st.button("Atualizar agora", key="refresh_now")

    if every and every > 0:
        st.fragment(run_every=float(every))(_render_content)(st, db_path, show_synthetic)
    else:
        _render_content(st, db_path, show_synthetic)


# -- entry points -------------------------------------------------------------------------


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="cnnlab.tcc.demo_streamlit", description=PAGE_TITLE)
    parser.add_argument("--db", default="", help="path of the TCC v2 SQLite database (read-only)")
    return parser.parse_known_args(argv)


def launch_command(db: str, streamlit_options: Sequence[str] = ()) -> list[str]:
    """`streamlit run` on this file with the same interpreter; page arguments go after `--`.

    Streamlit listens on every network interface by default; the page shows
    the whole experiment database, so it stays on this machine unless the
    operator passes `--server.address` explicitly.
    """
    options = list(streamlit_options)
    if not any(o == "--server.address" or o.startswith("--server.address=") for o in options):
        options = ["--server.address", LOCAL_ADDRESS, *options]
    command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *options, "--"]
    return command + (["--db", db] if db else [])


def main(argv: Sequence[str] | None = None) -> int:
    """Inside streamlit: draw the page. From plain Python: launch `streamlit run` on this file."""
    args, streamlit_options = _parse(list(sys.argv[1:] if argv is None else argv))
    try:
        import streamlit as st
        from streamlit import runtime
    except ModuleNotFoundError:
        print(
            f"streamlit is not installed in this environment. Install it with: {STREAMLIT_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 2
    if runtime.exists():
        render(st, args.db)
        return 0
    return subprocess.call(launch_command(args.db, streamlit_options))


if __name__ == "__main__":
    code = main()
    if code:
        raise SystemExit(code)
