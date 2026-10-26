"""Structured JSON logging: stderr lines plus a background shipper to Axiom.

Adapted from the Kokoro pod (tts_kokoro_api/app/axiom_handler.py).

Why Axiom: when the pod's processes die (segfault, OOM, hard exit), local
files inside the container can be lost on container recreation. Axiom holds
logs durably so crashes that leave nothing behind can still be examined.

The handler buffers records on a bounded in-memory queue and flushes from a
daemon thread every `flush_interval` seconds (or whenever the batch fills).
At most ~2 s of logs sit in memory, so even a hard crash loses only a few
records; the build-up before a crash is what matters.

Best-effort by design: Axiom outages must never block requests or crash the
app. Transport failures drop the batch silently. The handler does nothing if
AXIOM_TOKEN/AXIOM_DATASET aren't set.

Structured fields go in ``extra=``: ``log.info("synth", extra={"voice": ...})``
becomes top-level keys in both the stderr line and the Axiom event.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import error as _urlerr
from urllib import request as _urlreq

from app.config import SERVICE_NAME

_AXIOM_INGEST = "https://api.axiom.co/v1/datasets/{dataset}/ingest"
_FLUSH_INTERVAL_S = 2.0
_BATCH_SIZE = 100
_MAX_QUEUE = 10_000
_HTTP_TIMEOUT_S = 5.0

# Standard LogRecord attributes; anything else on the record is treated as
# structured "extra" data and emitted as a top-level field. Keep this list
# in sync with the cpython logging module (3.11+).
_RECORD_BUILTINS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


def record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """The record's ``extra=`` fields plus a formatted exception, if any."""
    fields: dict[str, Any] = {}
    if record.exc_info:
        fields["exc"] = "".join(traceback.format_exception(*record.exc_info))
    elif record.exc_text:
        fields["exc"] = record.exc_text
    for key, value in record.__dict__.items():
        if key in _RECORD_BUILTINS or key.startswith("_"):
            continue
        try:
            json.dumps(value)
            fields[key] = value
        except (TypeError, ValueError):
            fields[key] = repr(value)
    return fields


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, lvl, logger, msg, service, extras."""

    def __init__(self, service: str, static: dict[str, Any]) -> None:
        super().__init__()
        self._service = service
        self._static = static

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "service": self._service,
            **self._static,
        }
        event.update(record_fields(record))
        return json.dumps(event, ensure_ascii=False, default=repr)


class AxiomHandler(logging.Handler):
    def __init__(
        self,
        token: str,
        dataset: str,
        *,
        service: str = SERVICE_NAME,
        static: Optional[dict[str, Any]] = None,
        batch_size: int = _BATCH_SIZE,
        flush_interval: float = _FLUSH_INTERVAL_S,
        max_queue: int = _MAX_QUEUE,
    ) -> None:
        super().__init__()
        self._url = _AXIOM_INGEST.format(dataset=dataset)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._service = service
        self._static = dict(static or {})
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._loop, name="axiom-shipper", daemon=True
        )
        self._thread.start()
        atexit.register(self._shutdown)
        # SIGTERM is what Docker sends on container stop. Without an explicit
        # handler the default action exits without running atexit on some
        # interpreters; install a flush-on-term hook.
        try:
            signal.signal(signal.SIGTERM, self._on_signal)
        except (ValueError, OSError):
            # Not the main thread, or signal already handled; skip.
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = self._record_to_event(record)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped += 1
        except Exception:
            # Logging must never raise.
            pass

    def _record_to_event(self, record: logging.LogRecord) -> dict[str, Any]:
        event: dict[str, Any] = {
            "_time": int(record.created * 1000),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "service": self._service,
            "where": f"{record.module}.{record.funcName}",
            "lineno": record.lineno,
            **self._static,
        }
        for key, value in record_fields(record).items():
            event.setdefault(key, value)
        return event

    def _loop(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        while not self._stop.is_set():
            timeout = max(0.05, self._flush_interval - (time.monotonic() - last_flush))
            try:
                batch.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                pass
            now = time.monotonic()
            if batch and (
                len(batch) >= self._batch_size
                or now - last_flush >= self._flush_interval
            ):
                self._flush(batch)
                batch = []
                last_flush = now
        # Drain anything left in the queue on shutdown.
        try:
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        try:
            payload = json.dumps(batch, default=repr).encode("utf-8")
            req = _urlreq.Request(
                self._url, data=payload, headers=self._headers, method="POST"
            )
            with _urlreq.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                resp.read()
        except _urlerr.URLError:
            # Network/HTTP failure: drop the batch. Retries are not worth the
            # queue growth on a sustained outage.
            pass
        except Exception:
            pass

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self._shutdown()
        # Restore the default behavior so the process actually exits.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=3.0)


class _NotUvicornAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "uvicorn.access"


_installed: Optional[AxiomHandler] = None
_configured = False


def install(
    level: int = logging.INFO,
    *,
    service: str = SERVICE_NAME,
    static: Optional[dict[str, Any]] = None,
) -> Optional[AxiomHandler]:
    """Install the Axiom handler on the root logger if env is configured.

    Returns the handler instance, or None if AXIOM_TOKEN/AXIOM_DATASET are
    not set (silent no-op so dev environments without Axiom keys just work).
    """
    global _installed
    if _installed is not None:
        return _installed

    token = (os.environ.get("AXIOM_TOKEN") or "").strip()
    dataset = (os.environ.get("AXIOM_DATASET") or "").strip()
    if not (token and dataset):
        return None

    handler = AxiomHandler(token, dataset, service=service, static=static)
    handler.setLevel(level)
    # Uvicorn's per-request access log is high-volume and low-signal; the
    # app's own synth lines and every error go through.
    handler.addFilter(_NotUvicornAccess())

    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    _installed = handler
    return handler


def setup_logging(
    service: str = SERVICE_NAME, level: int = logging.INFO, **static: Any,
) -> None:
    """JSON lines on stderr plus Axiom shipping; idempotent per process.

    ``static`` fields (e.g. ``worker=2``) are added to every record.
    """
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(service, static))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    install(level, service=service, static=static)
    _configured = True
