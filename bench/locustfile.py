"""Locust load for the Ming TTS server: one corpus sentence per POST /v1/tts.

    MING_TOKEN=... locust -f bench/locustfile.py --host https://<podid>-8080.proxy.runpod.net

Besides the normal request stats, every success fires three custom entries so
Locust's UI shows their percentiles side by side:
  POST /v1/tts [SERVER_RUNTIME]  server-reported synthesis time (ms)
  POST /v1/tts [QUEUE]           server-reported queue wait (ms)
  POST /v1/tts [RTF x1000]       wall latency / audio seconds, x1000 (1000 = real time)
Read those rows individually; Locust's "Aggregated" row mixes them in.
"""

import os
import time
from itertools import cycle

from locust import HttpUser, constant, task

from corpus import CORPUS

ENDPOINT = "/v1/tts"
LANGUAGE = os.environ.get("MING_LANGUAGE", "en")
request_payload_base = {
    "voice": os.environ.get("MING_VOICE", "US_James"),
    "language": LANGUAGE,
    "sample_rate": int(os.environ.get("MING_SAMPLE_RATE", "24000")),
    "output_format": "pcm",
}
sentences = cycle(CORPUS[LANGUAGE])


class ListenerUser(HttpUser):
    # Closed loop like loadtest.py: each user asks for the next sentence as
    # soon as the previous one arrives (a real listener is paced by playback).
    wait_time = constant(0)

    def on_start(self):
        token = os.environ.get("MING_TOKEN")
        if not token:
            raise RuntimeError("set MING_TOKEN")
        self.client.headers["Authorization"] = f"Bearer {token}"

    @task
    def synthesize_sentence(self):
        payload = {**request_payload_base, "text": next(sentences)}

        start_time = time.perf_counter()
        with self.client.post(ENDPOINT, json=payload, catch_response=True, timeout=120) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000
            if response.status_code != 200:
                response.failure(f"{response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
                audio_s = float(body["duration_seconds"])
                status = body["inference_status"]
            except (ValueError, KeyError, TypeError) as exc:
                response.failure(f"invalid body: {exc}")
                return
            response.success()

        for name, value in (
            ("SERVER_RUNTIME", status.get("runtime_ms")),
            ("QUEUE", status.get("queue_ms")),
            ("RTF x1000", latency_ms / audio_s if audio_s > 0 else None),  # ms / s = RTF x 1000
        ):
            if value is None:
                continue
            self.environment.events.request.fire(
                request_type="POST",
                name=f"{ENDPOINT} [{name}]",
                response_time=value,
                response_length=0,
                exception=None,
                context={},
            )
