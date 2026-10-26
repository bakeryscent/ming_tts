"""The aligner process and its client in the API process.

Why one process: the aligner (MMS_FA, ~1.2 GB of VRAM in fp32) must not be
loaded once per synthesis worker, which share the GPU between 3-4 model
replicas; and the API process never imports torch. So a single extra
process holds the aligner and serves the whole pool, one job at a time
(a few seconds of audio align in tens of milliseconds on the GPU, far less
than a synthesis, so a serial aligner keeps up with every replica).

Why it can never fail a request: alignment only improves the timestamps.
Not ready, dead, timed out or erroring, the API falls back to the estimate
and says ``words_source: "estimated"``.

Protocol, pickled tuples over a duplex Pipe (same shape as app.worker):

    api -> aligner   ("align", job_id, payload)   payload: see AlignerService.align
                     ("stop",)
    aligner -> api   ("ready", info)              once, after load + warmup
                     ("result", job_id, spans)    spans: list per fragment, or None
                     ("error", job_id, message)
                     ("fatal", message)           loading failed; the process exits
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
import multiprocessing
import signal
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Sequence

from app import aligner as aligner_module
from app.axiom_handler import setup_logging
from app.config import SERVICE_NAME, Settings
from app.timestamps import Fragment
from app.worker import _die_with_parent

log = logging.getLogger("ming-tts-api.aligner")

WATCHDOG_INTERVAL_S = 1.0
MAX_RESPAWN_DELAY_S = 60.0


# -- the aligner process ---------------------------------------------------

def run(conn: Connection, settings: Settings) -> None:
    """Process target: load the aligner, report ready, then serve jobs."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _die_with_parent()
    setup_logging(SERVICE_NAME, worker="aligner")

    try:
        backend, info = aligner_module.load_backend(settings.aligner, settings.device)
    except Exception as error:
        log.exception("aligner_load_failed")
        try:
            conn.send(("fatal", repr(error)))
        except OSError:
            pass
        sys.exit(1)
    log.info("aligner_loaded", extra=info)

    try:
        conn.send(("ready", info))
        while True:
            message = conn.recv()
            if message[0] == "stop":
                return
            _, job_id, payload = message
            try:
                spans = aligner_module.align_fragments(
                    backend, payload["pcm"], payload["sample_rate"], payload["language"],
                    payload["fragments"])
                conn.send(("result", job_id, spans))
            except Exception as error:
                log.exception("alignment_failed", extra={"job": job_id})
                conn.send(("error", job_id, repr(error)))
    except (EOFError, OSError):
        return


# -- the client in the API process -----------------------------------------

@dataclass(eq=False)
class _Job:
    id: int
    payload: dict[str, Any]
    future: asyncio.Future
    started_at: float = 0.0


class AlignerService:
    """Owns the aligner process; ``align`` never raises, it returns ``None`` instead."""

    def __init__(
        self,
        settings: Settings,
        target: Callable[[Connection, Settings], None] = run,
    ) -> None:
        self._settings = settings
        self._target = target
        self._ctx = multiprocessing.get_context("spawn")
        self._pending: collections.deque[_Job] = collections.deque()
        self._job_ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watchdog: asyncio.Task | None = None
        self._stopping = False
        self.generation = 0
        self.state = "starting"  # starting | idle | busy | dead
        self.process: BaseProcess | None = None
        self.conn: Connection | None = None
        self.job: _Job | None = None
        self.failed_starts = 0
        self.restarts = 0
        self.jobs_done = 0
        self.fallbacks = 0  # requests answered with the estimate
        self.info: dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._spawn()
        self._watchdog = asyncio.create_task(self._watch())

    async def stop(self, timeout: float = 10.0) -> None:
        self._stopping = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        for job in list(self._pending) + ([self.job] if self.job else []):
            if not job.future.done():
                job.future.set_result(None)
        self._pending.clear()
        if self.conn is not None:
            try:
                self.conn.send(("stop",))
            except OSError:
                pass
        if self.process is not None:
            await asyncio.to_thread(_join, self.process, timeout)
        if self.conn is not None:
            self.conn.close()

    @property
    def ready(self) -> bool:
        return self.state in ("idle", "busy")

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self._settings.aligner,
            "state": self.state,
            "pid": self.process.pid if self.process is not None else None,
            "restarts": self.restarts,
            "jobs_done": self.jobs_done,
            "fallbacks": self.fallbacks,
            "queue": len(self._pending),
            "timeout_s": self._settings.aligner_timeout_s,
            **self.info,
        }

    # -- jobs ------------------------------------------------------------------
    async def align(
        self,
        pcm: bytes,
        sample_rate: int,
        language: str,
        fragments: Sequence[Fragment],
        words: Sequence[Sequence[str]],
    ) -> list[list[aligner_module.WordSpan] | None] | None:
        """Align every fragment's words; ``None`` when the aligner cannot answer.

        The deadline (``ALIGNER_TIMEOUT_S``) covers queueing and alignment;
        a late result is dropped. One entry per fragment comes back: the
        spans relative to the fragment, or ``None`` for a fragment that
        failed on its own.
        """
        if self._stopping or not self.ready or not aligner_module.supports(language):
            self.fallbacks += 1
            return None
        assert self._loop is not None
        payload = {
            "pcm": pcm, "sample_rate": sample_rate, "language": language,
            "fragments": [(fragment.offset, fragment.duration, list(fragment_words))
                          for fragment, fragment_words in zip(fragments, words)],
        }
        job = _Job(id=next(self._job_ids), payload=payload, future=self._loop.create_future())
        self._pending.append(job)
        self._dispatch()
        try:
            result = await asyncio.wait_for(job.future, self._settings.aligner_timeout_s)
        except asyncio.TimeoutError:
            self._discard(job)
            log.warning("alignment_timeout", extra={
                "job": job.id, "timeout_s": self._settings.aligner_timeout_s,
                "where": "running" if job.started_at else "queued"})
            result = None
        except asyncio.CancelledError:
            self._discard(job)
            raise
        if result is None:
            self.fallbacks += 1
        return result

    def _discard(self, job: _Job) -> None:
        try:
            self._pending.remove(job)
        except ValueError:
            pass  # running: the worker finishes it and the result is dropped

    def _dispatch(self) -> None:
        while self._pending and self.state == "idle":
            job = self._pending.popleft()
            if job.future.done():
                continue
            job.started_at = time.monotonic()
            self.job, self.state = job, "busy"
            try:
                assert self.conn is not None
                self.conn.send(("align", job.id, job.payload))
            except (OSError, ValueError) as error:
                self.job = None
                self._pending.appendleft(job)
                self._retire(f"send failed: {error!r}")

    # -- process -------------------------------------------------------------
    def _spawn(self) -> None:
        if self._stopping:
            return
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(target=self._target, args=(child_conn, self._settings),
                                    name="ming-aligner", daemon=True)
        try:
            process.start()
        except Exception:
            log.exception("aligner_spawn_failed")
            parent_conn.close()
            child_conn.close()
            self._schedule_respawn(failed=True)
            return
        child_conn.close()
        self.process, self.conn, self.state = process, parent_conn, "starting"
        threading.Thread(target=self._read, args=(self.generation, parent_conn),
                         name="ming-aligner-reader", daemon=True).start()
        log.info("aligner_spawned", extra={"pid": process.pid, "backend": self._settings.aligner})

    def _read(self, generation: int, conn: Connection) -> None:
        while True:
            try:
                message = conn.recv()
            except Exception:
                self._post(self._on_exit, generation)
                return
            self._post(self._on_message, generation, message)

    def _post(self, callback: Callable[..., None], *args: Any) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            pass

    def _on_message(self, generation: int, message: tuple) -> None:
        if generation != self.generation:
            return
        kind = message[0]
        if kind == "ready":
            self.state, self.failed_starts = "idle", 0
            self.info.update(message[1])
            log.info("aligner_ready", extra=message[1])
        elif kind in ("result", "error"):
            job, self.job, self.state = self.job, None, "idle"
            self.jobs_done += 1
            if job is None or job.id != message[1] or job.future.done():
                pass  # the request stopped waiting; drop the outcome
            elif kind == "result":
                job.future.set_result(message[2])
            else:
                log.warning("alignment_error", extra={"job": job.id, "error": message[2]})
                job.future.set_result(None)
        elif kind == "fatal":
            log.error("aligner_load_failed", extra={"error": message[1]})
        self._dispatch()

    def _on_exit(self, generation: int) -> None:
        if generation != self.generation or self._stopping:
            return
        self._retire("process exited")

    def _retire(self, reason: str) -> None:
        was_ready = self.ready
        process, job = self.process, self.job
        self.generation += 1
        self.state, self.job = "dead", None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        exitcode = None
        if process is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=1.0)
            exitcode = process.exitcode
        if job is not None and not job.future.done():
            job.future.set_result(None)
        # Jobs still queued would wait for a process that is not there yet.
        while self._pending:
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(None)
        self.restarts += 1
        log.error("aligner_died", extra={"reason": reason, "exitcode": exitcode,
                                         "restarts": self.restarts})
        self._schedule_respawn(failed=not was_ready)

    def _schedule_respawn(self, *, failed: bool) -> None:
        self.failed_starts = self.failed_starts + 1 if failed else 0
        delay = min(2.0 ** self.failed_starts, MAX_RESPAWN_DELAY_S) if failed else 0.0
        assert self._loop is not None
        self._loop.call_later(delay, self._spawn)

    async def _watch(self) -> None:
        """Kill the aligner when a job runs far past the request's deadline."""
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            job = self.job
            if self.state == "busy" and job is not None \
                    and time.monotonic() - job.started_at > self._settings.stuck_align_s:
                self._retire(f"job {job.id} hung for {time.monotonic() - job.started_at:.0f}s")


def _join(process: BaseProcess, timeout: float) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2.0)
    if process.is_alive():
        process.kill()
        process.join(1.0)
