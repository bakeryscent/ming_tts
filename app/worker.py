"""Worker process entry point: one model replica serving jobs from the pool.

Protocol: pickled tuples over a duplex ``multiprocessing`` Pipe.

    pool -> worker   ("synth", job_id, plan_payload)   run one job
                     ("stop",)                         exit cleanly
    worker -> pool   ("ready", info)                   once, after load + warmup
                     ("result", job_id, result)        result: see MingBackend.synthesize
                     ("error", job_id, kind, message)  kind: invalid | oom | error
                     ("fatal", message)                loading failed; the process exits

A worker handles one job at a time. It exits when the pipe closes, which is
how it notices that the API process has gone away.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from multiprocessing.connection import Connection
from typing import Any, Protocol

import numpy as np

from app.audio import float_to_pcm16
from app.axiom_handler import setup_logging
from app.config import SERVICE_NAME, Settings
from app.engine import GpuOutOfMemory, SynthesisPlan

log = logging.getLogger("ming-tts-api.worker")


class Backend(Protocol):
    def info(self) -> dict[str, Any]: ...

    def synthesize(self, plan: SynthesisPlan) -> dict[str, Any]: ...


class FakeBackend:
    """FAKE_ENGINE=1 stand-in for tests: no torch, a 220 Hz tone per request.

    FAKE_LOAD_S and FAKE_SYNTH_S (seconds) simulate model load and synthesis
    time; the tone lasts 60 ms per character of text.
    """

    def __init__(self) -> None:
        time.sleep(float(os.environ.get("FAKE_LOAD_S", "0.1")))
        self._synth_s = float(os.environ.get("FAKE_SYNTH_S", "0.05"))

    def info(self) -> dict[str, Any]:
        return {"gpu_name": "fake", "vram_total_mb": 0, "vram_used_mb": 0,
                "prompt_cache_size": 0, "load_s": 0.0}

    def synthesize(self, plan: SynthesisPlan) -> dict[str, Any]:
        started = time.perf_counter()
        time.sleep(self._synth_s)
        seconds = min(0.06 * len(plan.text), 60.0)
        t = np.arange(int(seconds * plan.sample_rate)) / plan.sample_rate
        return {
            "pcm": float_to_pcm16(0.3 * np.sin(2 * np.pi * 220 * t)),
            "sample_rate": plan.sample_rate,
            "fragments": [(plan.text, seconds)],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "prompt_cache_size": 0,
            "vram_used_mb": 0,
        }


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this worker when the API process dies.

    A replica orphaned by a crashed or SIGKILLed parent would keep its VRAM,
    and the restarted server's replicas would then run out of GPU memory.
    Linux only; elsewhere the worker still exits once its pipe closes.
    """
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    pr_set_pdeathsig = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(pr_set_pdeathsig, signal.SIGKILL)


def _quiet_ming_logs() -> None:
    """Ming logs every decoded prompt (including the caller's text) through
    loguru at INFO; keep only its warnings."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _load_backend(settings: Settings) -> tuple[Backend, dict[str, Any]]:
    if settings.fake_engine:
        backend: Backend = FakeBackend()
        return backend, backend.info()
    _quiet_ming_logs()
    from app.engine import MingBackend

    ming = MingBackend(settings)
    warmup_ms = ming.warmup()
    return ming, {**ming.info(), "warmup_ms": warmup_ms}


def run(index: int, conn: Connection, settings: Settings) -> None:
    """Process target: load the model, report ready, then serve jobs."""
    # Ctrl-C reaches the whole process group; let the API process decide
    # when workers stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _die_with_parent()
    setup_logging(SERVICE_NAME, worker=index)

    try:
        backend, info = _load_backend(settings)
    except Exception as error:
        log.exception("worker_load_failed")
        try:
            conn.send(("fatal", repr(error)))
        except OSError:
            pass
        sys.exit(1)
    log.info("worker_loaded", extra=info)

    try:
        conn.send(("ready", info))
        while True:
            message = conn.recv()
            if message[0] == "stop":
                return
            _, job_id, payload = message
            conn.send(_run_job(backend, job_id, payload))
    except (EOFError, OSError):
        # The API process closed the pipe or died: nothing left to serve.
        return


def _run_job(backend: Backend, job_id: int, payload: dict[str, Any]) -> tuple:
    try:
        return ("result", job_id, backend.synthesize(SynthesisPlan.from_payload(payload)))
    except GpuOutOfMemory as error:
        log.error("synthesis_oom", extra={"job": job_id})
        return ("error", job_id, "oom", str(error))
    except ValueError as error:
        return ("error", job_id, "invalid", str(error))
    except Exception as error:
        log.exception("synthesis_failed", extra={"job": job_id})
        return ("error", job_id, "error", f"synthesis failed: {error}")
