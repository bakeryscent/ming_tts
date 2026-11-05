"""HTTP API for the self-hosted Ming-3.0-tts server (contract: docs/API.md).

This process holds no model: it validates requests, routes them to a Ming
task (app.engine), queues them for the worker pool (app.pool) and shapes the
response. It never imports torch.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app import engine
from app.aligner import merge_alignment
from app.aligner_service import AlignerService
from app.audio import encode_audio, pcm_duration_seconds
from app.axiom_handler import setup_logging
from app.config import MAX_UPLOAD_BYTES, SERVICE_NAME, Settings
from app.errors import ApiError, PayloadTooLarge
from app.pool import WorkerPool
from app.timestamps import Fragment, estimate_words, fragment_tokens
from app.voices import VoiceRegistry

setup_logging(SERVICE_NAME)
log = logging.getLogger(SERVICE_NAME)

LATENCY_WINDOW = 1000


class WordOut(BaseModel):
    id: int
    start: float
    end: float
    text: str


class InferenceStatus(BaseModel):
    status: str = "succeeded"
    cost: float = 0.0
    runtime_ms: int
    queue_ms: int
    tokens_input: int
    fragments: int
    worker: int
    # Milliseconds spent in forced alignment (None when it did not run).
    align_ms: int | None = None


class TTSResponse(BaseModel):
    audio: str
    output_format: str
    sample_rate: int
    duration_seconds: float
    words: list[WordOut]
    words_source: str | None
    inference_status: InferenceStatus


@dataclass
class Metrics:
    """Request counters for /stats; latencies cover the last LATENCY_WINDOW successes."""

    started_at: float = field(default_factory=time.time)
    requests: int = 0
    succeeded: int = 0
    failures: Counter = field(default_factory=Counter)  # keyed by status_for()
    audio_seconds: float = 0.0
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))

    @staticmethod
    def status_for(error: ApiError) -> str:
        return {429: "rejected", 503: "not_ready", 504: "timeout"}.get(
            error.status_code, "error" if error.status_code >= 500 else "invalid")

    def record_failure(self, status: str) -> None:
        self.failures[status] += 1

    def record_success(self, latency_ms: float, audio_s: float) -> None:
        self.succeeded += 1
        self.audio_seconds += audio_s
        self.latencies_ms.append(latency_ms)

    def _percentile(self, fraction: float) -> int | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        return int(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))])

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": int(time.time() - self.started_at),
            "requests": self.requests, "succeeded": self.succeeded,
            "invalid": self.failures["invalid"], "errors": self.failures["error"],
            "timeouts": self.failures["timeout"], "rejected": self.failures["rejected"],
            "not_ready": self.failures["not_ready"],
            "audio_seconds": round(self.audio_seconds, 1),
            "latency_ms": {"p50": self._percentile(0.50), "p95": self._percentile(0.95),
                           "window": len(self.latencies_ms)},
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = VoiceRegistry(settings.voices_dir, settings.bundled_voices_dir)
    # Under FAKE_ENGINE the Ming checkout is optional (tests without the repo).
    tts_tasks = engine.load_tts_tasks(settings.ming_dir, required=not settings.fake_engine)
    pool = WorkerPool(settings)
    aligner = AlignerService(settings) if settings.aligner != "off" else None
    metrics = Metrics()

    async def timestamps(result, plan: engine.SynthesisPlan) -> tuple[list[dict], str, int | None]:
        """``(words, words_source, align_ms)`` for one synthesis result.

        Aligned when the aligner answered in time for every fragment;
        otherwise the estimate (app.timestamps), possibly for some
        fragments only, labelled ``"estimated"``.
        """
        fragments, offset = [], 0.0
        for text, seconds in result.fragments:
            fragments.append(Fragment(text=text, offset=offset, duration=seconds))
            offset += seconds
        if aligner is None:
            return estimate_words(fragments, source_text=plan.text), "estimated", None
        started = time.monotonic()
        words = [[token.text for token in tokens]
                 for tokens in fragment_tokens(fragments, source_text=plan.text)]
        spans = await aligner.align(result.pcm, result.sample_rate, plan.language, fragments, words)
        align_ms = int((time.monotonic() - started) * 1000)
        merged, fully_aligned = merge_alignment(fragments, plan.text, spans)
        return merged, "aligned" if fully_aligned else "estimated", align_ms

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        seeded = registry.seed()
        log.info("startup", extra={
            "workers": settings.num_workers, "max_queue": settings.max_queue,
            "request_timeout_s": settings.request_timeout_s, "model_id": settings.model_id,
            "voices_dir": str(settings.voices_dir), "voices_seeded": seeded,
            "fake_engine": settings.fake_engine, "aligner": settings.aligner})
        await pool.start()
        if aligner is not None:
            await aligner.start()
        yield
        if aligner is not None:
            await aligner.stop()
        await pool.stop()

    app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["*"], allow_headers=["*"], max_age=600)

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse({"detail": error.detail}, status_code=error.status_code,
                            headers=error.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        messages = []
        for item in error.errors():
            location = ".".join(str(part) for part in item.get("loc", ())[1:])
            messages.append(f"{location}: {item.get('msg')}" if location else str(item.get("msg")))
        return JSONResponse({"detail": "; ".join(messages)}, status_code=400)

    bearer = HTTPBearer(auto_error=False)
    token = settings.api_token.encode()

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if creds is None or not secrets.compare_digest(creds.credentials.encode(), token):
            raise HTTPException(status_code=401, detail="Invalid token",
                                headers={"WWW-Authenticate": "Bearer"})

    auth = [Depends(require_auth)]

    @app.get("/health")
    async def health() -> JSONResponse:
        body = {
            "ok": pool.healthy,
            "workers": {"total": settings.num_workers, "busy": pool.busy_count,
                        "ready": pool.ready_count},
            "queue": pool.queue_depth,
            "model_loaded": pool.started,
        }
        return JSONResponse(body, status_code=200 if pool.healthy else 503)

    @app.get("/stats", dependencies=auth)
    async def stats() -> dict[str, Any]:
        workers = pool.snapshot()
        details = workers["workers"]
        gpu = next((w for w in details if w.get("gpu_name")), {})
        return {
            **metrics.snapshot(),
            "queue": workers["queue"],
            "max_queue": workers["max_queue"],
            "workers": workers,
            "prompt_cache_size": sum(w.get("prompt_cache_size", 0) for w in details),
            "clone_voices": registry.clone_count(),
            "aligner": aligner.snapshot() if aligner is not None else {"backend": "off"},
            "gpu": {
                "name": gpu.get("gpu_name"),
                "vram_total_mb": gpu.get("vram_total_mb"),
                # Device-wide figure as last reported by any worker.
                "vram_used_mb": max((w.get("vram_used_mb", 0) for w in details), default=0),
            },
        }

    @app.get("/v1/voices", dependencies=auth)
    async def list_voices() -> dict[str, Any]:
        return {"voices": [voice.public() for voice in registry.list()]}

    @app.post("/v1/voices/clone", dependencies=auth)
    async def clone_voice(
        audio: UploadFile = File(...),
        prompt_text: str = Form(...),
        language: str = Form("auto"),
        voice_id: str | None = Form(None, alias="id"),
    ) -> dict[str, Any]:
        data = await audio.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise PayloadTooLarge(f"audio upload is limited to {MAX_UPLOAD_BYTES // 2**20} MB")
        voice = await asyncio.to_thread(
            registry.create_clone, data, prompt_text, language, (voice_id or "").strip() or None)
        log.info("voice_cloned", extra={"voice": voice.id, "language": voice.language,
                                        "prompt_seconds": voice.prompt_seconds})
        return voice.public()

    @app.delete("/v1/voices/{voice_id}", dependencies=auth)
    async def delete_voice(voice_id: str) -> dict[str, Any]:
        registry.delete(voice_id)
        log.info("voice_deleted", extra={"voice": voice_id})
        return {"ok": True, "id": voice_id}

    @app.post("/v1/tts", dependencies=auth, response_model=TTSResponse)
    async def tts(req: engine.TTSRequest) -> TTSResponse:
        started = time.monotonic()
        metrics.requests += 1
        plan: engine.SynthesisPlan | None = None
        try:
            plan = engine.plan_request(req, registry, tts_tasks, settings.max_text_chars)
            result = await pool.submit(plan.to_payload())
        except ApiError as error:
            status = metrics.status_for(error)
            metrics.record_failure(status)
            log.warning("synth", extra={
                "status": status, "voice": req.voice, "task": plan.task if plan else None,
                "chars": len(req.text), "detail": error.detail,
                "latency_ms": int((time.monotonic() - started) * 1000)})
            raise

        duration = pcm_duration_seconds(result.pcm, result.sample_rate)
        words: list[dict] = []
        words_source: str | None = None
        align_ms: int | None = None
        if req.return_timestamps:
            words, words_source, align_ms = await timestamps(result, plan)
        latency_ms = (time.monotonic() - started) * 1000
        metrics.record_success(latency_ms, duration)
        log.info("synth", extra={
            "status": "succeeded", "voice": plan.voice, "task": plan.task,
            "language": plan.language, "chars": len(plan.text),
            "fragments": len(result.fragments), "audio_s": round(duration, 3),
            "queue_ms": result.queue_ms, "runtime_ms": result.runtime_ms,
            "align_ms": align_ms, "words_source": words_source,
            "latency_ms": int(latency_ms),
            "rtf": round(result.runtime_ms / 1000 / duration, 3) if duration else None,
            "worker": result.worker})

        return TTSResponse(
            audio=encode_audio(result.pcm, result.sample_rate, req.output_format),
            output_format=req.output_format,
            sample_rate=result.sample_rate,
            duration_seconds=round(duration, 3),
            words=[WordOut(**word) for word in words],
            # "aligned" (forced alignment, ALIGNER=mms_fa) or "estimated"
            # (spread over the audio by word length, app.timestamps).
            words_source=words_source,
            inference_status=InferenceStatus(
                runtime_ms=result.runtime_ms, queue_ms=result.queue_ms,
                tokens_input=len(plan.text), fragments=len(result.fragments),
                worker=result.worker, align_ms=align_ms),
        )

    return app


app = create_app()
