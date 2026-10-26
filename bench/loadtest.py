#!/usr/bin/env python3
"""Capacity sweep for the Ming TTS server (POST /v1/tts).

At each concurrency level `c`, `c` closed-loop clients send one corpus sentence
per request, back to back, over keep-alive connections. Per level it reports
latency percentiles, real-time factor (RTF = wall latency / audio seconds),
aggregate throughput in audio seconds per wall second, and admission errors.
The final block turns the best level into an estimate of concurrent listeners
and a cost per 1000 audio seconds; see README.md for the model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any

import httpx

from corpus import CORPUS

TTS_PATH = "/v1/tts"
DEFAULT_LEVELS = "1,2,3,4,6,8,12,16"
DEFAULT_RETRY_AFTER_S = 2.0
DURATION_MISMATCH_S = 0.05
# A level only counts as "throughput still growing" if it beats the best
# level so far by this much; otherwise extra concurrency is just queueing.
GROWTH_EPSILON = 0.05
PROGRESS_EVERY_S = 30.0


@dataclass(frozen=True)
class Settings:
    url: str
    token: str
    voice: str = "US_James"
    language: str = "en"
    sample_rate: int = 24000
    timeout_s: float = 120.0
    rtf_target: float = 0.8
    max_error_rate: float = 0.05
    early_stop: bool = True
    gpu_hourly_usd: float = 0.16


@dataclass
class RequestRecord:
    chars: int
    status: int  # final HTTP status; 0 = transport error or client-side timeout
    first_status: int  # 429 here means admission was refused, even if the retry succeeded
    latency_s: float  # first send -> final response, including any Retry-After wait
    started_s: float  # offset from the level start
    audio_s: float = 0.0
    pcm_audio_s: float = 0.0
    runtime_ms: float | None = None
    queue_ms: float | None = None
    worker: int | None = None
    fragments: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def failure_label(self) -> str:
        if self.status == 0:
            return "transport"
        if self.status == 200:
            return "invalid-body"
        return str(self.status)


@dataclass
class LevelSummary:
    concurrency: int
    n: int
    ok: int
    failed: int
    rejected_429: int  # requests whose first attempt got 429
    retried_ok: int  # ...of which the single retry succeeded
    first_try_errors: int  # requests not served cleanly on the first attempt
    error_rate: float
    failures: dict[str, int]
    wall_s: float
    latency_mean_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    rtf_mean: float | None
    rtf_p95: float | None
    audio_s_total: float
    audio_s_mean: float | None
    throughput_audio_s_per_s: float
    chars_per_s: float
    requests_per_s: float
    queue_ms_mean: float | None
    server_rtf_mean: float | None
    littles_in_flight: float | None  # throughput x mean latency / mean audio; ~c in a healthy closed loop
    max_in_flight: int
    workers: dict[str, int]
    duration_mismatches: int  # |duration_seconds - PCM length| > DURATION_MISMATCH_S
    records: list[RequestRecord] = field(repr=False)


@dataclass
class Capacity:
    concurrency: int
    throughput_audio_s_per_s: float
    listeners: int
    rtf_p95: float
    headroom: float  # 1 - p95 RTF: share of a sentence's playback time still unused at p95
    in_flight_per_listener: float  # mean latency / mean audio duration
    peaked_below: int | None  # highest passing level, when it no longer raised throughput
    usd_per_1k_audio_s: float
    usd_per_1m_chars: float


@dataclass
class SweepResult:
    levels: list[LevelSummary]
    stopped: str | None
    capacity: Capacity | None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_levels(spec: str) -> list[int]:
    levels = [int(part) for part in spec.split(",") if part.strip()]
    if not levels or any(c < 1 for c in levels):
        raise ValueError(f"concurrency levels must be positive integers: {spec!r}")
    return levels


def make_client(
    settings: Settings, max_connections: int, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.token}"},
        timeout=httpx.Timeout(settings.timeout_s),
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        transport=transport,
    )


def _retry_after_s(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_S)))
    except ValueError:
        return DEFAULT_RETRY_AFTER_S


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:200]
    except (ValueError, AttributeError):
        return response.text[:200]


def _fill_from_body(record: RequestRecord, response: httpx.Response, default_sample_rate: int) -> RequestRecord:
    try:
        body = response.json()
        pcm = base64.b64decode(body["audio"], validate=True)
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        record.error = f"invalid body: {exc}"
        return record
    sample_rate = body.get("sample_rate") or default_sample_rate
    record.pcm_audio_s = len(pcm) / 2 / sample_rate  # s16le mono
    record.audio_s = float(body.get("duration_seconds") or record.pcm_audio_s)
    if record.audio_s <= 0:
        record.error = "empty audio"
        return record
    status = body.get("inference_status") or {}
    record.runtime_ms = status.get("runtime_ms")
    record.queue_ms = status.get("queue_ms")
    record.worker = status.get("worker")
    record.fragments = status.get("fragments")
    return record


async def synthesize(client: httpx.AsyncClient, settings: Settings, text: str, level_start: float) -> RequestRecord:
    """One logical request; a 429 is retried once after Retry-After."""
    payload = {
        "text": text,
        "voice": settings.voice,
        "language": settings.language,
        "sample_rate": settings.sample_rate,
        "output_format": "pcm",
    }
    started = time.perf_counter()
    first_status: int | None = None
    while True:
        try:
            response = await client.post(TTS_PATH, json=payload)
        except httpx.HTTPError as exc:
            return RequestRecord(
                chars=len(text),
                status=0,
                first_status=first_status if first_status is not None else 0,
                latency_s=time.perf_counter() - started,
                started_s=started - level_start,
                error=f"{type(exc).__name__}: {exc}",
            )
        if first_status is None:
            first_status = response.status_code
            if response.status_code == 429:
                await asyncio.sleep(_retry_after_s(response))
                continue
        break
    record = RequestRecord(
        chars=len(text),
        status=response.status_code,
        first_status=first_status,
        latency_s=time.perf_counter() - started,
        started_s=started - level_start,
    )
    if response.status_code != 200:
        record.error = _error_detail(response)
        return record
    return _fill_from_body(record, response, settings.sample_rate)


def summarize_level(
    concurrency: int, records: list[RequestRecord], wall_s: float, max_in_flight: int
) -> LevelSummary:
    ok = [r for r in records if r.ok]
    latencies = [r.latency_s for r in ok]
    rtfs = [r.latency_s / r.audio_s for r in ok]
    audio_total = sum(r.audio_s for r in ok)
    latency_mean = mean(latencies)
    audio_mean = mean([r.audio_s for r in ok])
    throughput = audio_total / wall_s if wall_s > 0 else 0.0
    rejected = [r for r in records if r.first_status == 429]
    first_try_errors = sum(1 for r in records if not r.ok or r.first_status != 200)
    queue = [float(r.queue_ms) for r in ok if r.queue_ms is not None]
    server_rtfs = [r.runtime_ms / 1000 / r.audio_s for r in ok if r.runtime_ms is not None]
    workers = Counter(str(r.worker) for r in ok if r.worker is not None)
    return LevelSummary(
        concurrency=concurrency,
        n=len(records),
        ok=len(ok),
        failed=len(records) - len(ok),
        rejected_429=len(rejected),
        retried_ok=sum(1 for r in rejected if r.ok),
        first_try_errors=first_try_errors,
        error_rate=first_try_errors / len(records) if records else 0.0,
        failures=dict(sorted(Counter(r.failure_label for r in records if not r.ok).items())),
        wall_s=wall_s,
        latency_mean_s=latency_mean,
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        latency_p99_s=percentile(latencies, 99),
        rtf_mean=mean(rtfs),
        rtf_p95=percentile(rtfs, 95),
        audio_s_total=audio_total,
        audio_s_mean=audio_mean,
        throughput_audio_s_per_s=throughput,
        chars_per_s=sum(r.chars for r in ok) / wall_s if wall_s > 0 else 0.0,
        requests_per_s=len(ok) / wall_s if wall_s > 0 else 0.0,
        queue_ms_mean=mean(queue),
        server_rtf_mean=mean(server_rtfs),
        littles_in_flight=throughput * latency_mean / audio_mean if latency_mean and audio_mean else None,
        max_in_flight=max_in_flight,
        workers=dict(sorted(workers.items())),
        duration_mismatches=sum(1 for r in ok if abs(r.audio_s - r.pcm_audio_s) > DURATION_MISMATCH_S),
        records=records,
    )


async def _report_progress(concurrency: int, records: list[RequestRecord], start: float) -> None:
    while True:
        await asyncio.sleep(PROGRESS_EVERY_S)
        ok = sum(1 for r in records if r.ok)
        elapsed = time.perf_counter() - start
        print(f"  c={concurrency} t={elapsed:.0f}s done={len(records)} ok={ok}", file=sys.stderr, flush=True)


async def run_level(
    client: httpx.AsyncClient,
    settings: Settings,
    sentences: Iterator[str],
    concurrency: int,
    *,
    requests: int | None = None,
    duration_s: float | None = None,
) -> LevelSummary:
    """Closed loop: `concurrency` clients each send the next request as soon as
    the previous one returns, until `requests` are issued or `duration_s` has
    elapsed (requests already in flight at the deadline still count)."""
    if (requests is None) == (duration_s is None):
        raise ValueError("pass exactly one of requests / duration_s")
    records: list[RequestRecord] = []
    issued = in_flight = max_in_flight = 0
    start = time.perf_counter()
    deadline = start + duration_s if duration_s is not None else math.inf

    def claim() -> bool:
        nonlocal issued
        if (requests is not None and issued >= requests) or time.perf_counter() >= deadline:
            return False
        issued += 1
        return True

    async def worker_loop() -> None:
        nonlocal in_flight, max_in_flight
        while claim():
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                records.append(await synthesize(client, settings, next(sentences), start))
            finally:
                in_flight -= 1

    progress = asyncio.create_task(_report_progress(concurrency, records, start)) if duration_s else None
    try:
        await asyncio.gather(*(worker_loop() for _ in range(concurrency)))
    finally:
        if progress:
            progress.cancel()
    return summarize_level(concurrency, records, time.perf_counter() - start, max_in_flight)


def meets_target(summary: LevelSummary, rtf_target: float) -> bool:
    return (
        summary.ok > 0
        and summary.first_try_errors == 0
        and summary.rtf_p95 is not None
        and summary.rtf_p95 <= rtf_target
    )


def stop_reason(summary: LevelSummary, settings: Settings) -> str | None:
    if summary.error_rate > settings.max_error_rate:
        return f"error rate {summary.error_rate:.1%} > {settings.max_error_rate:.0%}"
    if summary.rtf_p95 is None or summary.rtf_p95 > settings.rtf_target:
        p95 = "n/a" if summary.rtf_p95 is None else f"{summary.rtf_p95:.2f}"
        return f"p95 RTF {p95} > target {settings.rtf_target:.2f}"
    return None


def estimate_capacity(levels: Sequence[LevelSummary], settings: Settings) -> Capacity | None:
    """Highest concurrency that still grows throughput while meeting the RTF target.

    A listener consumes 1 audio second per wall second, so the pod can feed as
    many continuous listeners as audio seconds it produces per second.
    """
    passing = sorted((s for s in levels if meets_target(s, settings.rtf_target)), key=lambda s: s.concurrency)
    if not passing:
        return None
    best = passing[0]
    for summary in passing[1:]:
        if summary.throughput_audio_s_per_s > best.throughput_audio_s_per_s * (1 + GROWTH_EPSILON):
            best = summary
    assert best.rtf_p95 is not None and best.latency_mean_s is not None and best.audio_s_mean is not None
    audio_rate = best.throughput_audio_s_per_s
    usd_per_s = settings.gpu_hourly_usd / 3600
    return Capacity(
        concurrency=best.concurrency,
        throughput_audio_s_per_s=audio_rate,
        listeners=math.floor(audio_rate),
        rtf_p95=best.rtf_p95,
        headroom=1 - best.rtf_p95,
        in_flight_per_listener=best.latency_mean_s / best.audio_s_mean,
        peaked_below=passing[-1].concurrency if passing[-1] is not best else None,
        usd_per_1k_audio_s=usd_per_s / audio_rate * 1000,
        usd_per_1m_chars=usd_per_s / best.chars_per_s * 1_000_000,
    )


async def run_sweep(
    client: httpx.AsyncClient,
    settings: Settings,
    levels: Sequence[int],
    sentences: Iterator[str],
    *,
    requests: int | None = None,
    duration_s: float | None = None,
    on_level: Callable[[LevelSummary], None] | None = None,
) -> SweepResult:
    summaries: list[LevelSummary] = []
    stopped: str | None = None
    for concurrency in levels:
        summary = await run_level(client, settings, sentences, concurrency, requests=requests, duration_s=duration_s)
        summaries.append(summary)
        if on_level:
            on_level(summary)
        reason = stop_reason(summary, settings)
        if reason and settings.early_stop:
            stopped = f"stopped after c={concurrency}: {reason}"
            break
    return SweepResult(levels=summaries, stopped=stopped, capacity=estimate_capacity(summaries, settings))


# --- rendering -------------------------------------------------------------

TABLE_HEADER = (
    "| c | n | ok | 429 raw / retried-ok | other err | p50 s | p95 s | p99 s | RTF mean | RTF p95 "
    "| audio s/s | chars/s | queue ms | srv RTF | in-flight max | workers |\n"
    "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"
)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table_row(s: LevelSummary) -> str:
    other_errors = sum(count for label, count in s.failures.items() if label != "429")
    workers = " ".join(f"{w}:{n}" for w, n in s.workers.items()) or "-"
    return (
        f"| {s.concurrency} | {s.n} | {s.ok} | {s.rejected_429} / {s.retried_ok} | {other_errors} "
        f"| {_fmt(s.latency_p50_s)} | {_fmt(s.latency_p95_s)} | {_fmt(s.latency_p99_s)} "
        f"| {_fmt(s.rtf_mean)} | {_fmt(s.rtf_p95)} | {_fmt(s.throughput_audio_s_per_s)} "
        f"| {_fmt(s.chars_per_s, 1)} | {_fmt(s.queue_ms_mean, 0)} | {_fmt(s.server_rtf_mean)} "
        f"| {s.max_in_flight} | {workers} |"
    )


def capacity_lines(result: SweepResult, settings: Settings) -> list[str]:
    cap = result.capacity
    if cap is None:
        return [
            f"No level met p95 RTF <= {settings.rtf_target:.2f} with zero 429/5xx; "
            "the pod cannot keep even the lowest level gap-free."
        ]
    lines = [
        f"- Best level: **c={cap.concurrency}** (p95 RTF {cap.rtf_p95:.2f} <= target {settings.rtf_target:.2f}, "
        "no 429/5xx)",
        f"- Throughput: **{cap.throughput_audio_s_per_s:.2f} audio s / wall s**",
        f"- Estimated concurrent listeners: **~{cap.listeners}** (each listener plays 1 audio s per s; "
        f"each keeps ~{cap.in_flight_per_listener:.2f} requests in flight)",
        f"- Headroom: at p95 a sentence is ready within {cap.rtf_p95:.0%} of its own playback time "
        f"({cap.headroom:.0%} slack to absorb bursts, seeks and session starts)",
        f"- Cost at ${settings.gpu_hourly_usd:.2f}/h: **${cap.usd_per_1k_audio_s:.4f} per 1000 audio s**, "
        f"**${cap.usd_per_1m_chars:.2f} per 1M characters**",
    ]
    if cap.peaked_below is not None:
        lines.append(
            f"- Throughput stopped growing after c={cap.concurrency}; levels up to c={cap.peaked_below} still met "
            "the target but only added queueing."
        )
    return lines


def render_markdown(result: SweepResult, settings: Settings, mode: str, health: dict[str, Any] | None) -> str:
    workers = (health or {}).get("workers", {}).get("total", "?")
    parts = [
        "# Ming TTS capacity sweep",
        "",
        f"- Server: `{settings.url}` (workers: {workers})",
        f"- Voice `{settings.voice}`, language `{settings.language}`, {settings.sample_rate} Hz, {mode}",
        f"- RTF target (p95): {settings.rtf_target:.2f}; early stop: {'on' if settings.early_stop else 'off'}",
        "",
        TABLE_HEADER,
        *(table_row(s) for s in result.levels),
        "",
    ]
    if result.stopped:
        parts += [f"_{result.stopped}_", ""]
    parts += ["## Capacity estimate", "", *capacity_lines(result, settings), ""]
    return "\n".join(parts)


def result_json(
    result: SweepResult,
    settings: Settings,
    health: dict[str, Any] | None,
    stats_after: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "settings": {k: v for k, v in asdict(settings).items() if k != "token"},
        "health": health,
        "stats_after": stats_after,
        "stopped": result.stopped,
        "capacity": asdict(result.capacity) if result.capacity else None,
        "levels": [asdict(s) for s in result.levels],
    }


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("MING_URL"), help="server base URL (env MING_URL)")
    p.add_argument("--token", default=os.environ.get("MING_TOKEN"), help="bearer token (env MING_TOKEN)")
    p.add_argument("--voice", default="US_James")
    p.add_argument("--language", default="en", choices=sorted(CORPUS))
    p.add_argument("--sample-rate", type=int, default=24000, choices=(24000, 44100))
    p.add_argument("--concurrency", default=DEFAULT_LEVELS, help=f"comma-separated levels (default {DEFAULT_LEVELS})")
    load = p.add_mutually_exclusive_group()
    load.add_argument("--requests", type=int, default=24, help="requests per level (default 24)")
    load.add_argument("--duration", type=float, help="sustained mode: seconds to hold each level")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    p.add_argument("--warmup", type=int, default=6, help="unmeasured requests before the sweep (0 to skip)")
    p.add_argument("--rtf-target", type=float, default=0.8, help="max acceptable p95 RTF (default 0.8)")
    p.add_argument("--max-error-rate", type=float, default=0.05, help="early-stop error rate (default 0.05)")
    p.add_argument("--no-early-stop", action="store_true", help="run every level even past the target")
    p.add_argument("--gpu-hourly", type=float, default=0.16, help="GPU price in USD/h for the cost line")
    p.add_argument("--out", type=Path, default=Path("results.json"))
    p.add_argument("--md", type=Path, default=Path("results.md"))
    return p


async def _get_json(client: httpx.AsyncClient, path: str) -> tuple[int, dict[str, Any] | None]:
    try:
        response = await client.get(path)
        return response.status_code, response.json()
    except (httpx.HTTPError, ValueError):
        return 0, None


async def main_async(args: argparse.Namespace) -> int:
    if not args.url or not args.token:
        print("error: --url/--token (or MING_URL/MING_TOKEN) are required", file=sys.stderr)
        return 2
    settings = Settings(
        url=args.url,
        token=args.token,
        voice=args.voice,
        language=args.language,
        sample_rate=args.sample_rate,
        timeout_s=args.timeout,
        rtf_target=args.rtf_target,
        max_error_rate=args.max_error_rate,
        early_stop=not args.no_early_stop,
        gpu_hourly_usd=args.gpu_hourly,
    )
    levels = parse_levels(args.concurrency)
    requests = None if args.duration else args.requests
    mode = f"{args.duration:.0f} s per level" if args.duration else f"{args.requests} requests per level"
    sentences = cycle(CORPUS[args.language])

    async with make_client(settings, max(levels)) as client:
        status, health = await _get_json(client, "/health")
        if status != 200:
            print(f"error: /health returned {status or 'no response'}: {health}", file=sys.stderr)
            return 1
        print(f"server {settings.url} health: {json.dumps(health)}", file=sys.stderr)

        if args.warmup:
            warm = await run_level(client, settings, sentences, min(args.warmup, max(levels)), requests=args.warmup)
            print(
                f"warmup: {warm.ok}/{warm.n} ok, p50 {_fmt(warm.latency_p50_s)} s, RTF mean {_fmt(warm.rtf_mean)}",
                file=sys.stderr,
            )

        print(TABLE_HEADER, flush=True)
        result = await run_sweep(
            client,
            settings,
            levels,
            sentences,
            requests=requests,
            duration_s=args.duration,
            on_level=lambda s: print(table_row(s), flush=True),
        )
        _, stats_after = await _get_json(client, "/stats")

    markdown = render_markdown(result, settings, mode, health)
    print()
    if result.stopped:
        print(f"_{result.stopped}_\n")
    print("\n".join(["## Capacity estimate", "", *capacity_lines(result, settings)]))
    args.out.write_text(json.dumps(result_json(result, settings, health, stats_after), indent=2))
    args.md.write_text(markdown)
    print(f"\nwrote {args.out} and {args.md}", file=sys.stderr)
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async(build_parser().parse_args())))


if __name__ == "__main__":
    main()
