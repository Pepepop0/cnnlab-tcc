"""Consome a fila e executa os jobs.

Requisito central: uma excecao num job **nunca** derruba o loop. E isso que faz
o modo Sleep existir — a sessao da madrugada tem que sobreviver a um job que
estoura OOM ou a um disco que encheu.

Este modulo e o dono do evento terminal da run. O core sinaliza cancelamento
pelo campo `cancelled` do resultado e para; quem traduz isso em `RunCancelled`
e em status de fila e aqui (ver docs/DECISIONS.md).
"""

from __future__ import annotations

import gc
import threading
import traceback
from collections.abc import Callable

import torch

from cnnlab.config import DatasetConfig, ModelConfig
from cnnlab.core.data import DataBundle, load_data
from cnnlab.core.ga.engine import run_ga
from cnnlab.core.ga.genome import to_train_config
from cnnlab.core.trainer import UnitContext, train
from cnnlab.events import Event, RunCancelled, RunFailed, RunFinished, RunStarted
from cnnlab.storage.layout import new_run_id
from cnnlab.storage.writer import RunWriter
from cnnlab.worker.queue import Job, JobQueue

# Espera da fila vazia. Longo o bastante para nao consumir CPU a noite toda,
# curto o bastante para um job novo comecar sem o usuario perceber.
_IDLE_WAIT_S = 2.0


class Runner:
    """Executor sequencial de jobs. Uma instancia por worker thread."""

    def __init__(
        self,
        queue: JobQueue,
        on_event: Callable[[Event], None],
        device: torch.device,
    ) -> None:
        self.queue = queue
        self.on_event = on_event
        self.device = torch.device(device)

    def run_forever(self, stop: threading.Event, cancel_current: threading.Event) -> None:
        """Loop do modo Sleep.

        `stop` encerra o loop depois do job atual. `cancel_current` aborta so o
        job em execucao — a fila continua, por isso ele e limpo a cada volta.
        """
        while not stop.is_set():
            job = self.queue.next_pending()
            if job is None:
                stop.wait(_IDLE_WAIT_S)
                continue

            try:
                self.queue.mark(job.id, "running")
                self.run_one(job, cancel_current)
            except Exception:
                # Ultima linha de defesa: falha antes de haver `RunWriter`
                # (pasta que nao pode ser criada, fila ilegivel) nao pode
                # derrubar o loop. `run_one` ja trata o que acontece dentro.
                self._mark_quietly(job.id, "failed")
            finally:
                cancel_current.clear()
                self._release_memory()

    def run_one(self, job: Job, cancel: threading.Event) -> None:
        """Executa um job do inicio ao fim. Nunca propaga excecao de execucao."""
        spec = job.spec
        run_id = new_run_id(spec.kind)
        writer: RunWriter | None = None

        def callback(evt: Event) -> None:
            if writer is not None:
                writer.handle(evt)
            self.on_event(evt)

        try:
            writer = RunWriter(run_id, spec)
            # A run aparece como `running` ja durante a carga do dataset, que
            # na primeira vez inclui o download do CIFAR-100.
            writer.start()

            data = self._load_data(spec.dataset)
            if spec.kind == "classic":
                self._run_classic(job, run_id, writer, data, callback, cancel)
            else:
                self._run_ga(job, run_id, writer, data, callback, cancel)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if writer is not None:
                writer.fail(exc)
            callback(RunFailed(run_id=run_id, error=error, traceback=traceback.format_exc()))
            self._mark_quietly(job.id, "failed", run_id)

    # -- modos -----------------------------------------------------------

    def _run_classic(
        self,
        job: Job,
        run_id: str,
        writer: RunWriter,
        data: DataBundle,
        callback: Callable[[Event], None],
        cancel: threading.Event,
    ) -> None:
        """Treino unico. Aqui o runner emite `RunStarted` e `RunFinished`."""
        cfg = job.spec.train
        callback(
            RunStarted(
                run_id=run_id,
                kind="classic",
                unit_total=1,
                epochs_per_unit=cfg.epochs,
            )
        )

        ctx = UnitContext(run_id=run_id, unit_index=0, unit_total=1)
        result = train(cfg, data, ctx, callback, cancel, self.device, keep_weights=True)

        # Pesos de uma run cancelada sao pesos de verdade, das epocas que
        # completaram. Salvar custa nada e evita jogar fora GPU ja gasta.
        artifact = self._save_artifact(writer, result.state_dict, cfg.model, data.class_names, result.final_val_acc)

        if result.cancelled:
            self._finish_cancelled(job, run_id, callback, at_unit=ctx.unit_index)
            return

        callback(
            RunFinished(
                run_id=run_id,
                kind="classic",
                best_fitness=result.best_val_acc,
                best_genome=None,
                total_trainings=1,
                total_epochs=result.epochs_run,
                elapsed_s=result.elapsed_s,
                artifact=artifact,
            )
        )
        self._mark_quietly(job.id, "done", run_id)

    def _run_ga(
        self,
        job: Job,
        run_id: str,
        writer: RunWriter,
        data: DataBundle,
        callback: Callable[[Event], None],
        cancel: threading.Event,
    ) -> None:
        """Busca evolutiva. `run_ga` ja emite `RunStarted` e `RunFinished`."""
        result = run_ga(job.spec, data, run_id, callback, cancel, self.device)

        if result.cancelled:
            # Sem retreino final, entao `best_state_dict` e None: nao ha
            # artefato para salvar.
            self._finish_cancelled(job, run_id, callback, at_unit=result.total_trainings)
            return

        model_cfg = to_train_config(
            result.best_genome,
            epochs=job.spec.ga.final_epochs,
            seed=job.spec.ga.seed,
            amp=job.spec.amp,
            channels_last=job.spec.channels_last,
        ).model
        # `final_val_acc` e a acuracia do retreino final, que e o que esta
        # salvo. `best_fitness` mediria as poucas epocas da busca e
        # subestimaria o modelo.
        val_acc = result.final_val_acc if result.final_val_acc is not None else result.best_fitness
        self._save_artifact(writer, result.best_state_dict, model_cfg, data.class_names, val_acc)
        self._mark_quietly(job.id, "done", run_id)

    # -- auxiliares ------------------------------------------------------

    def _load_data(self, cfg: DatasetConfig) -> DataBundle:
        """Carrega o dataset.

        `core.data.load_data` mantem cache de modulo com chave no dump da
        config, entao jobs consecutivos com a mesma `DatasetConfig` reusam os
        tensores ja na VRAM. Recarregar o CIFAR-100 a cada job custaria dezenas
        de segundos por job durante a noite.
        """
        return load_data(cfg, self.device)

    def _save_artifact(
        self,
        writer: RunWriter,
        state_dict: dict | None,
        model_cfg: ModelConfig,
        class_names: list[str],
        val_acc: float,
    ) -> str | None:
        """Grava `best.pt` se houver pesos. Devolve o nome para o `RunFinished`."""
        if not state_dict:
            return None
        return writer.save_artifact(state_dict, model_cfg, class_names, val_acc).name

    def _finish_cancelled(
        self,
        job: Job,
        run_id: str,
        callback: Callable[[Event], None],
        at_unit: int,
    ) -> None:
        """Fecha uma run cancelada.

        `RunCancelled` e o evento terminal — nunca `RunFinished`. Consequencia
        aceita: `RunWriter` so escreve `summary.json` a partir de `RunFinished`,
        entao run cancelada nao tem `summary.json`. O historico ainda a lista,
        pelo `status.json`, com `best_fitness` e `elapsed_s` vazios. Emitir
        `RunFinished` para ganhar o summary sobrescreveria o status para `done`
        e mentiria sobre o que aconteceu.
        """
        callback(RunCancelled(run_id=run_id, at_unit=at_unit))
        self._mark_quietly(job.id, "cancelled", run_id)

    def _mark_quietly(self, job_id: str, status: str, run_id: str | None = None) -> None:
        """Marca o job na fila sem deixar erro de disco derrubar o loop."""
        try:
            self.queue.mark(job_id, status, run_id)
        except Exception:
            pass

    def _release_memory(self) -> None:
        """Devolve VRAM entre jobs. O cache do dataset sobrevive de proposito."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
