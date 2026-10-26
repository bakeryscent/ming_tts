"""Pool of model worker processes and the job queue in front of them.

Why processes: Ming's decode loop is bound by Python overhead and kernel
launches rather than GPU compute, and MingTTS assumes batch size 1. Separate
processes, each with its own MingTTS and CUDA context, decode concurrently on
one GPU without sharing a GIL. The API process never imports torch.

Why a small queue with early 429s: callers (the Bun router) have other
providers to fall back to. Telling them "busy" at once beats letting a
request wait behind a long queue and time out anyway.

Threading model: everything here runs on the asyncio loop except one reader
thread per worker, which blocks on the pipe and hands every message (or the
pipe's EOF, meaning the worker died) to the loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Callable

from app import worker
from app.config import Settings
from app.errors import BadRequest, NotReady, QueueFull, RequestTimeout, SynthesisError

log = logging.getLogger("ming-tts-api.pool")

WATCHDOG_INTERVAL_S = 1.0
MAX_RESPAWN_DELAY_S = 60.0


@dataclass(frozen=True)
class JobResult:
    pcm: bytes
    sample_rate: int
    fragments: list[tuple[str, float]]  # (text, seconds), in playback order
    runtime_ms: int
    queue_ms: int
    worker: int


@dataclass(eq=False)
class _Job:
    id: int
    payload: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float
    started_at: float = 0.0


@dataclass(eq=False)
class _Slot:
    index: int
    # Bumped whenever the slot's process is retired, so messages still in
    # flight from a dead process's reader thread are recognised and dropped.
    generation: int = 0
    state: str = "starting"  # starting | idle | busy | dead
    process: BaseProcess | None = None
    conn: Connection | None = None
    job: _Job | None = None
    ready_once: bool = False
    failed_starts: int = 0  # consecutive processes that died before "ready"
    restarts: int = 0
    jobs_done: int = 0
    info: dict[str, Any] = field(default_factory=dict)


class WorkerPool:
    def __init__(
        self,
        settings: Settings,
        target: Callable[[int, Connection, Settings], None] = worker.run,
    ) -> None:
        self._settings = settings
        self._target = target
        # "spawn": CUDA cannot be used in a forked child of a process that
        # touched it, and spawned workers start from a clean interpreter.
        self._ctx = multiprocessing.get_context("spawn")
        self._slots = [_Slot(index=i) for i in range(settings.num_workers)]
        self._pending: collections.deque[_Job] = collections.deque()
        self._job_ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watchdog: asyncio.Task | None = None
        self._stopping = False
        self._started = False
        self.respawns = 0

    # -- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        """Spawn every worker and return; workers report "ready" when loaded."""
        self._loop = asyncio.get_running_loop()
        for slot in self._slots:
            self._spawn(slot)
        self._watchdog = asyncio.create_task(self._watch())

    async def stop(self, timeout: float = 10.0) -> None:
        self._stopping = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        while self._pending:
            job = self._pending.popleft()
            if not job.future.done():
                job.future.set_exception(NotReady("server is shutting down"))
        for slot in self._slots:
            if slot.job is not None and not slot.job.future.done():
                slot.job.future.set_exception(NotReady("server is shutting down"))
            if slot.conn is not None:
                try:
                    slot.conn.send(("stop",))
                except OSError:
                    pass
        processes = [slot.process for slot in self._slots if slot.process is not None]
        await asyncio.to_thread(_join_all, processes, timeout)
        for slot in self._slots:
            if slot.conn is not None:
                slot.conn.close()

    # -- public state ----------------------------------------------------------
    @property
    def started(self) -> bool:
        """True once every worker has loaded the model and warmed up."""
        return self._started

    @property
    def healthy(self) -> bool:
        # After startup a respawning replica only lowers capacity; the pod
        # stays healthy while at least one replica can serve.
        return self._started and not self._stopping and self.ready_count > 0

    @property
    def ready_count(self) -> int:
        return sum(1 for slot in self._slots if slot.state in ("idle", "busy"))

    @property
    def busy_count(self) -> int:
        return sum(1 for slot in self._slots if slot.state == "busy")

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    def snapshot(self) -> dict[str, Any]:
        workers = []
        for slot in self._slots:
            workers.append({
                "index": slot.index,
                "state": slot.state,
                "pid": slot.process.pid if slot.process is not None else None,
                "restarts": slot.restarts,
                "jobs_done": slot.jobs_done,
                **slot.info,
            })
        return {
            "total": len(self._slots),
            "ready": self.ready_count,
            "busy": self.busy_count,
            "respawns": self.respawns,
            "queue": self.queue_depth,
            "max_queue": self._settings.max_queue,
            "started": self._started,
            "workers": workers,
        }

    # -- jobs ------------------------------------------------------------------
    async def submit(self, payload: dict[str, Any]) -> JobResult:
        """Run one synthesis job; the deadline covers queueing and synthesis.

        Raises NotReady (503), QueueFull (429), RequestTimeout (504),
        BadRequest (400, input the model rejected) or SynthesisError (500).
        """
        if self._stopping:
            raise NotReady("server is shutting down")
        if not self._started:
            raise NotReady("workers are still loading the model")
        if len(self._pending) >= self._settings.max_queue and not self._idle_slot():
            raise QueueFull(f"all {len(self._slots)} workers busy and "
                            f"{len(self._pending)} requests queued")
        assert self._loop is not None
        job = _Job(id=next(self._job_ids), payload=payload,
                   future=self._loop.create_future(), enqueued_at=time.monotonic())
        self._pending.append(job)
        self._dispatch()
        try:
            return await asyncio.wait_for(job.future, self._settings.request_timeout_s)
        except asyncio.TimeoutError:
            self._discard(job)
            where = "queued" if not job.started_at else "synthesizing"
            raise RequestTimeout(
                f"request timed out after {self._settings.request_timeout_s:g}s ({where})"
            ) from None
        except asyncio.CancelledError:
            self._discard(job)
            raise

    def _discard(self, job: _Job) -> None:
        """Forget a job whose caller gave up.

        A running job cannot be interrupted: its worker stays busy until the
        synthesis ends and the result is dropped (the future is done).
        """
        try:
            self._pending.remove(job)
        except ValueError:
            pass

    def _idle_slot(self) -> _Slot | None:
        return next((slot for slot in self._slots if slot.state == "idle"), None)

    def _dispatch(self) -> None:
        while self._pending:
            slot = self._idle_slot()
            if slot is None:
                return
            job = self._pending.popleft()
            if job.future.done():
                continue
            job.started_at = time.monotonic()
            slot.job, slot.state = job, "busy"
            try:
                assert slot.conn is not None
                slot.conn.send(("synth", job.id, job.payload))
            except (OSError, ValueError) as error:
                # The worker is gone; requeue the job at the front.
                slot.job = None
                self._pending.appendleft(job)
                self._retire(slot, f"send failed: {error!r}")

    # -- worker processes ----------------------------------------------------
    def _spawn(self, slot: _Slot) -> None:
        if self._stopping:
            return
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=self._target, args=(slot.index, child_conn, self._settings),
            name=f"ming-worker-{slot.index}", daemon=True)
        try:
            process.start()
        except Exception:
            log.exception("worker_spawn_failed", extra={"worker": slot.index})
            parent_conn.close()
            child_conn.close()
            self._schedule_respawn(slot, failed=True)
            return
        # Close the parent's copy of the child end so the worker's death
        # shows up as EOF on parent_conn.
        child_conn.close()
        slot.process, slot.conn, slot.state = process, parent_conn, "starting"
        threading.Thread(
            target=self._read, args=(slot.index, slot.generation, parent_conn),
            name=f"ming-worker-{slot.index}-reader", daemon=True).start()
        log.info("worker_spawned", extra={"worker": slot.index, "pid": process.pid})

    def _read(self, index: int, generation: int, conn: Connection) -> None:
        """Reader thread: forward every message from one worker to the loop."""
        while True:
            try:
                message = conn.recv()
            except Exception:  # EOF, closed pipe or an unreadable message
                self._post(self._on_exit, index, generation)
                return
            self._post(self._on_message, index, generation, message)

    def _post(self, callback: Callable[..., None], *args: Any) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _on_message(self, index: int, generation: int, message: tuple) -> None:
        slot = self._slots[index]
        if generation != slot.generation:
            return
        kind = message[0]
        if kind == "ready":
            slot.state, slot.ready_once, slot.failed_starts = "idle", True, 0
            slot.info.update(message[1])
            log.info("worker_ready", extra={"worker": index, **message[1]})
            if not self._started and all(s.ready_once for s in self._slots):
                self._started = True
                log.info("pool_ready", extra={"workers": len(self._slots)})
        elif kind in ("result", "error"):
            job, slot.job, slot.state = slot.job, None, "idle"
            slot.jobs_done += 1
            if kind == "result":
                result = message[2]
                slot.info["prompt_cache_size"] = result["prompt_cache_size"]
                slot.info["vram_used_mb"] = result["vram_used_mb"]
            if job is None or job.id != message[1] or job.future.done():
                pass  # the caller timed out or went away; drop the outcome
            elif kind == "result":
                job.future.set_result(JobResult(
                    pcm=result["pcm"], sample_rate=result["sample_rate"],
                    fragments=[tuple(f) for f in result["fragments"]],
                    runtime_ms=result["runtime_ms"],
                    queue_ms=int((job.started_at - job.enqueued_at) * 1000),
                    worker=index))
            else:
                _, _, error_kind, detail = message
                job.future.set_exception(
                    BadRequest(detail) if error_kind == "invalid" else SynthesisError(detail))
        elif kind == "fatal":
            log.error("worker_load_failed", extra={"worker": index, "error": message[1]})
        self._dispatch()

    def _on_exit(self, index: int, generation: int) -> None:
        slot = self._slots[index]
        if generation != slot.generation or self._stopping:
            return
        self._retire(slot, "process exited")

    def _retire(self, slot: _Slot, reason: str) -> None:
        """Kill a slot's process, fail its job and schedule a replacement."""
        was_ready = slot.state in ("idle", "busy")
        process, job = slot.process, slot.job
        slot.generation += 1
        slot.state, slot.job = "dead", None
        if slot.conn is not None:
            slot.conn.close()
            slot.conn = None
        exitcode = None
        if process is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=1.0)
            exitcode = process.exitcode
        if job is not None and not job.future.done():
            job.future.set_exception(
                SynthesisError(f"worker {slot.index} died during synthesis"))
        self.respawns += 1
        slot.restarts += 1
        log.error("worker_died", extra={"worker": slot.index, "reason": reason,
                                        "exitcode": exitcode, "respawns": self.respawns})
        self._schedule_respawn(slot, failed=not was_ready)

    def _schedule_respawn(self, slot: _Slot, *, failed: bool) -> None:
        # Back off only while replicas die before ever becoming ready (bad
        # model path, OOM at load): a crash loop must not hammer the GPU.
        slot.failed_starts = slot.failed_starts + 1 if failed else 0
        delay = min(2.0 ** slot.failed_starts, MAX_RESPAWN_DELAY_S) if failed else 0.0
        assert self._loop is not None
        self._loop.call_later(delay, self._spawn, slot)

    async def _watch(self) -> None:
        """Kill workers whose job runs far past the deadline (hung CUDA call)."""
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            now = time.monotonic()
            for slot in self._slots:
                job = slot.job
                if slot.state == "busy" and job is not None \
                        and now - job.started_at > self._settings.stuck_job_s:
                    self._retire(slot, f"job {job.id} hung for {now - job.started_at:.0f}s")


def _join_all(processes: list[BaseProcess], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
