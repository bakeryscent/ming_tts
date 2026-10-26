"""Measurement-core tests against an in-process fake of POST /v1/tts."""

from __future__ import annotations

import asyncio
import base64
import math
from dataclasses import dataclass
from itertools import count, cycle

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import loadtest
from corpus import CORPUS
from loadtest import RequestRecord, Settings

SAMPLE_RATE = 24000


@dataclass
class FakeServer:
    """`workers` concurrent synth slots (FIFO behind them), each taking
    `service_s` per request and returning `audio_s` of silence; more than
    `max_in_flight` concurrent requests get 429."""

    audio_s: float = 0.5
    service_s: float = 0.02
    workers: int = 4
    max_in_flight: int | None = None
    retry_after: str = "0.01"
    in_flight: int = 0
    responses_429: int = 0

    def app(self) -> FastAPI:
        app = FastAPI()
        slots = asyncio.Semaphore(self.workers)
        worker_ids = count()
        pcm = base64.b64encode(bytes(int(self.audio_s * SAMPLE_RATE) * 2)).decode()

        @app.post("/v1/tts")
        async def tts(body: dict) -> JSONResponse:
            if self.max_in_flight is not None and self.in_flight >= self.max_in_flight:
                self.responses_429 += 1
                return JSONResponse(
                    {"detail": "queue full"}, status_code=429, headers={"Retry-After": self.retry_after}
                )
            self.in_flight += 1
            try:
                queued = asyncio.get_running_loop().time()
                async with slots:
                    queue_ms = (asyncio.get_running_loop().time() - queued) * 1000
                    await asyncio.sleep(self.service_s)
            finally:
                self.in_flight -= 1
            return JSONResponse(
                {
                    "audio": pcm,
                    "output_format": "pcm",
                    "sample_rate": body["sample_rate"],
                    "duration_seconds": self.audio_s,
                    "inference_status": {
                        "status": "succeeded",
                        "runtime_ms": self.service_s * 1000,
                        "queue_ms": queue_ms,
                        "fragments": 1,
                        "worker": next(worker_ids) % self.workers,
                    },
                }
            )

        return app


def settings(**overrides: object) -> Settings:
    return Settings(url="http://fake", token="test-token", **overrides)  # type: ignore[arg-type]


def client_for(server: FakeServer, max_connections: int = 16) -> httpx.AsyncClient:
    return loadtest.make_client(settings(), max_connections, transport=httpx.ASGITransport(app=server.app()))


def record(latency_s: float, audio_s: float, *, status: int = 200, first_status: int = 200, **extra: object) -> RequestRecord:
    return RequestRecord(
        chars=100,
        status=status,
        first_status=first_status,
        latency_s=latency_s,
        started_s=0.0,
        audio_s=audio_s,
        pcm_audio_s=audio_s,
        **extra,  # type: ignore[arg-type]
    )


def test_percentile_linear_interpolation() -> None:
    values = [float(v) for v in range(1, 101)]
    assert loadtest.percentile(values, 50) == pytest.approx(50.5)
    assert loadtest.percentile(values, 95) == pytest.approx(95.05)
    assert loadtest.percentile([3.0], 99) == 3.0
    assert loadtest.percentile([], 50) is None


def test_parse_levels() -> None:
    assert loadtest.parse_levels("1, 2,4,") == [1, 2, 4]
    with pytest.raises(ValueError):
        loadtest.parse_levels("0,2")


def test_summary_math_and_429_accounting() -> None:
    records = [
        record(1.0, 2.0, runtime_ms=800, queue_ms=100, worker=0),
        record(1.5, 3.0, runtime_ms=1200, queue_ms=200, worker=1),
        record(2.0, 2.0, first_status=429, runtime_ms=1000, queue_ms=0, worker=0),  # retried ok
        record(2.1, 0.0, status=429, first_status=429, error="queue full"),  # retry rejected too
        record(0.3, 0.0, status=500, first_status=500, error="boom"),
    ]
    s = loadtest.summarize_level(3, records, wall_s=4.0, max_in_flight=3)

    assert (s.n, s.ok, s.failed) == (5, 3, 2)
    assert (s.rejected_429, s.retried_ok) == (2, 1)
    assert s.first_try_errors == 3
    assert s.error_rate == pytest.approx(3 / 5)
    assert s.failures == {"429": 1, "500": 1}
    # Only successful requests feed latency/RTF/throughput.
    assert s.rtf_mean == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert s.latency_p50_s == pytest.approx(1.5)
    assert s.audio_s_total == pytest.approx(7.0)
    assert s.throughput_audio_s_per_s == pytest.approx(7.0 / 4.0)
    assert s.chars_per_s == pytest.approx(300 / 4.0)
    assert s.queue_ms_mean == pytest.approx(100.0)
    assert s.server_rtf_mean == pytest.approx((0.4 + 0.4 + 0.5) / 3)
    assert s.workers == {"0": 2, "1": 1}
    # Little's law: (7/4 audio s/s) * (4.5/3 s) / (7/3 audio s) = 1.125 requests in flight.
    assert s.littles_in_flight == pytest.approx(1.125)
    assert not loadtest.meets_target(s, 0.8)


def test_alignment_accounting() -> None:
    records = [
        record(1.0, 2.0, align_ms=40, words_source="aligned"),
        record(1.0, 2.0, align_ms=60, words_source="aligned"),
        record(1.0, 2.0, align_ms=500, words_source="estimated"),  # timed out: fell back
        record(1.0, 2.0),  # return_timestamps=false or ALIGNER=off
        record(0.3, 0.0, status=500, first_status=500, error="boom", align_ms=9, words_source="aligned"),
    ]
    s = loadtest.summarize_level(2, records, wall_s=2.0, max_in_flight=2)
    assert s.align_ms_mean == pytest.approx(200.0)
    assert s.aligned == 2
    assert "| 200 | 2/4 |" in loadtest.table_row(s)
    without = loadtest.summarize_level(1, [record(1.0, 2.0)], wall_s=1.0, max_in_flight=1)
    assert without.align_ms_mean is None and without.aligned == 0
    assert "| - | 0/1 |" in loadtest.table_row(without)


def test_capacity_estimate_and_cost() -> None:
    def level(c: int, latency_s: float, wall_s: float) -> loadtest.LevelSummary:
        return loadtest.summarize_level(c, [record(latency_s, 2.0) for _ in range(10)], wall_s, c)

    levels = [level(1, 0.5, 5.0), level(2, 0.6, 2.5), level(4, 1.2, 2.45), level(8, 2.0, 2.4)]
    cap = loadtest.estimate_capacity(levels, settings(rtf_target=0.8, gpu_hourly_usd=0.36))

    # c=4 (+2%) and c=8 (RTF 1.0, fails) do not beat c=2's 8 audio s/s meaningfully.
    assert cap is not None and cap.concurrency == 2
    assert cap.throughput_audio_s_per_s == pytest.approx(8.0)
    assert cap.listeners == 8
    assert cap.rtf_p95 == pytest.approx(0.3)
    assert cap.headroom == pytest.approx(0.7)
    assert cap.peaked_below == 4
    assert cap.usd_per_1k_audio_s == pytest.approx(0.36 / 3600 / 8.0 * 1000)
    assert cap.usd_per_1m_chars == pytest.approx(0.36 / 3600 / 400.0 * 1e6)
    assert loadtest.estimate_capacity([levels[-1]], settings()) is None


def test_level_against_fake_server() -> None:
    server = FakeServer(audio_s=0.5, service_s=0.02, workers=4)

    async def go() -> loadtest.LevelSummary:
        async with client_for(server) as client:
            return await loadtest.run_level(client, settings(), cycle(CORPUS["en"]), 4, requests=16)

    s = asyncio.run(go())
    assert (s.n, s.ok, s.rejected_429) == (16, 16, 0)
    assert s.max_in_flight == 4
    assert sum(s.workers.values()) == 16
    assert s.duration_mismatches == 0
    assert all(r.pcm_audio_s == pytest.approx(0.5) for r in s.records)
    assert s.server_rtf_mean == pytest.approx(0.04)
    assert 0.04 <= s.rtf_p95 < 0.2  # type: ignore[operator]
    # Ideal: 4 slots x 0.5 audio s / 0.02 s = 100 audio s/s; allow in-process overhead.
    assert 40 < s.throughput_audio_s_per_s <= 105
    assert s.littles_in_flight == pytest.approx(4, rel=0.35)


def test_429_retry_accounting_matches_server() -> None:
    server = FakeServer(audio_s=0.5, service_s=0.03, workers=2, max_in_flight=2, retry_after="0.02")

    async def go() -> loadtest.LevelSummary:
        async with client_for(server) as client:
            return await loadtest.run_level(client, settings(), cycle(CORPUS["en"]), 6, requests=30)

    s = asyncio.run(go())
    assert s.n == 30 and s.ok + s.failed == s.n
    assert s.rejected_429 > 0
    # Every rejected request cost one 429, plus one more if its retry was rejected too.
    assert server.responses_429 == s.rejected_429 + (s.rejected_429 - s.retried_ok)
    assert s.failures.get("429", 0) == s.rejected_429 - s.retried_ok
    assert s.first_try_errors >= s.rejected_429
    assert not loadtest.meets_target(s, 10.0)


def test_full_queue_stops_sweep_on_errors() -> None:
    server = FakeServer(max_in_flight=0, retry_after="0")

    async def go() -> loadtest.SweepResult:
        async with client_for(server) as client:
            return await loadtest.run_sweep(client, settings(), [1, 2], cycle(CORPUS["en"]), requests=5)

    result = asyncio.run(go())
    assert [s.concurrency for s in result.levels] == [1]
    only = result.levels[0]
    assert (only.ok, only.rejected_429, only.retried_ok) == (0, 5, 0)
    assert only.failures == {"429": 5}
    assert server.responses_429 == 10
    assert result.stopped is not None and "error rate" in result.stopped
    assert result.capacity is None


def sweep_saturating(early_stop: bool) -> loadtest.SweepResult:
    # 2 synth slots, 0.05 s each, 0.25 s of audio: RTF ~0.2 at c<=2, ~0.4 at c=4, ~0.8+ at c=8.
    server = FakeServer(audio_s=0.25, service_s=0.05, workers=2)

    async def go() -> loadtest.SweepResult:
        async with client_for(server) as client:
            return await loadtest.run_sweep(
                client,
                settings(rtf_target=0.5, early_stop=early_stop),
                [1, 2, 4, 8, 16],
                cycle(CORPUS["en"]),
                requests=12,
            )

    return asyncio.run(go())


def test_early_stop_on_rtf_and_capacity_pick() -> None:
    result = sweep_saturating(early_stop=True)
    assert [s.concurrency for s in result.levels] == [1, 2, 4, 8]
    assert result.stopped is not None and "p95 RTF" in result.stopped
    cap = result.capacity
    assert cap is not None
    # Throughput doubles from c=1 to c=2 (both slots busy) and is flat after.
    assert cap.concurrency == 2
    assert cap.throughput_audio_s_per_s == pytest.approx(10.0, rel=0.25)
    assert cap.listeners == math.floor(cap.throughput_audio_s_per_s)
    assert cap.peaked_below == 4

    markdown = loadtest.render_markdown(result, settings(rtf_target=0.5), "12 requests per level", None)
    assert "## Capacity estimate" in markdown and "c=2" in markdown
    assert markdown.count("\n| ") == 1 + len(result.levels)  # header + one row per level


def test_no_early_stop_runs_every_level() -> None:
    result = sweep_saturating(early_stop=False)
    assert [s.concurrency for s in result.levels] == [1, 2, 4, 8, 16]
    assert result.stopped is None


def test_corpus_shape() -> None:
    assert len(CORPUS["en"]) >= 60
    assert all(40 <= len(s) <= 250 for s in CORPUS["en"])
    assert len(set(CORPUS["en"])) == len(CORPUS["en"])
    assert all(len(CORPUS[lang]) == 5 for lang in ("es", "fr", "de", "zh"))
