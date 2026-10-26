"""Worker pool + HTTP layer, end to end with FAKE_ENGINE workers (no torch).

FAKE_LOAD_S / FAKE_SYNTH_S make the spawned workers slow on purpose, so
queueing, timeouts and crashes can be observed.
"""

from __future__ import annotations

import base64
import io
import os
import signal
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import ming_dir_or_none
from tests.helpers import ROOT, TOKEN, auth, wait_for, wav_bytes


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clients: list[TestClient] = []

    def _make(*, load_s: float = 0.1, synth_s: float = 0.05, **overrides) -> TestClient:
        # Spawned workers inherit the environment at spawn time.
        monkeypatch.setenv("FAKE_LOAD_S", str(load_s))
        monkeypatch.setenv("FAKE_SYNTH_S", str(synth_s))
        settings = Settings(
            api_token=TOKEN, voices_dir=tmp_path / "voices", bundled_voices_dir=ROOT / "voices",
            ming_dir=ming_dir_or_none() or tmp_path / "no-ming", fake_engine=True,
            **{"num_workers": 2, **overrides})
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


def wait_healthy(client: TestClient) -> dict:
    return wait_for(lambda: (r := client.get("/health")).status_code == 200 and r.json())


def tts(client: TestClient, **body):
    return client.post("/v1/tts", json={"voice": "US_James", **body}, headers=auth())


def parallel(calls):
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        return list(executor.map(lambda call: call(), calls))


def test_health_is_503_until_every_worker_is_ready(make_client):
    client = make_client(load_s=1.5, num_workers=2)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"ok": False, "workers": {"total": 2, "busy": 0, "ready": 0},
                               "queue": 0, "model_loaded": False}
    assert tts(client, text="Too early.").status_code == 503
    body = wait_healthy(client)
    assert body == {"ok": True, "workers": {"total": 2, "busy": 0, "ready": 2},
                    "queue": 0, "model_loaded": True}


def test_tts_response_shape(make_client):
    client = make_client()
    wait_healthy(client)
    response = tts(client, text="The garden will remain open until sunset.")
    assert response.status_code == 200, response.text
    body = response.json()
    pcm = base64.b64decode(body["audio"])
    assert body["output_format"] == "pcm" and body["sample_rate"] == 24000
    assert body["duration_seconds"] == pytest.approx(len(pcm) / 2 / 24000, abs=0.001)
    assert body["words_source"] == "estimated"
    assert [w["text"] for w in body["words"]] == [
        "The", "garden", "will", "remain", "open", "until", "sunset."]
    status = body["inference_status"]
    assert status["status"] == "succeeded" and status["fragments"] == 1
    assert status["tokens_input"] == 41 and status["worker"] in (0, 1)
    assert status["runtime_ms"] >= 40 and status["queue_ms"] >= 0


def test_wav_output_and_no_timestamps(make_client):
    client = make_client()
    wait_healthy(client)
    response = tts(client, text="Hello there.", output_format="wav", sample_rate=44100,
                   return_timestamps=False)
    body = response.json()
    with wave.open(io.BytesIO(base64.b64decode(body["audio"]))) as reader:
        assert reader.getframerate() == 44100 and reader.getnchannels() == 1
        assert reader.getnframes() / 44100 == pytest.approx(body["duration_seconds"], abs=0.001)
    assert body["words"] == [] and body["words_source"] is None


def test_requests_are_spread_over_all_workers(make_client):
    client = make_client(num_workers=3, synth_s=0.4)
    wait_healthy(client)
    responses = parallel([lambda i=i: tts(client, text=f"Request number {i}.") for i in range(6)])
    assert [r.status_code for r in responses] == [200] * 6
    assert {r.json()["inference_status"]["worker"] for r in responses} == {0, 1, 2}
    # Six jobs on three workers: the second wave waited for the first.
    assert max(r.json()["inference_status"]["queue_ms"] for r in responses) >= 200
    stats = client.get("/stats", headers=auth()).json()
    assert stats["succeeded"] == 6 and stats["requests"] == 6
    assert stats["latency_ms"]["window"] == 6 and stats["latency_ms"]["p95"] >= 400
    assert sum(w["jobs_done"] for w in stats["workers"]["workers"]) == 6


def test_queue_full_is_429(make_client):
    client = make_client(num_workers=1, max_queue=1, synth_s=1.0)
    wait_healthy(client)
    responses = parallel([lambda i=i: tts(client, text=f"Request {i}.") for i in range(3)])
    assert sorted(r.status_code for r in responses) == [200, 200, 429]
    rejected = next(r for r in responses if r.status_code == 429)
    assert rejected.headers["retry-after"] == "2"
    assert "queued" in rejected.json()["detail"]
    assert client.get("/stats", headers=auth()).json()["rejected"] == 1


def test_timeout_is_504(make_client):
    client = make_client(num_workers=1, synth_s=1.5, request_timeout_s=0.3)
    wait_healthy(client)
    response = tts(client, text="This takes too long.")
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]
    stats = client.get("/stats", headers=auth()).json()
    assert stats["timeouts"] == 1 and stats["workers"]["busy"] == 1
    # The worker finishes the abandoned job, drops its result and is free again.
    wait_for(lambda: client.get("/stats", headers=auth()).json()["workers"]["busy"] == 0)


def test_crashed_worker_is_respawned(make_client):
    client = make_client(num_workers=2, synth_s=2.0)
    wait_healthy(client)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(tts, client, text="Crash during this one.")
        busy = wait_for(lambda: [w for w in client.get("/stats", headers=auth())
                                 .json()["workers"]["workers"] if w["state"] == "busy"])
        os.kill(busy[0]["pid"], signal.SIGKILL)
        response = pending.result(timeout=10)
    assert response.status_code == 500
    assert "died" in response.json()["detail"]

    def respawned():
        workers = client.get("/stats", headers=auth()).json()["workers"]
        return workers["respawns"] == 1 and workers["ready"] == 2 and workers
    workers = wait_for(respawned)
    crashed = workers["workers"][busy[0]["index"]]
    assert crashed["restarts"] == 1 and crashed["pid"] != busy[0]["pid"]
    assert client.get("/health").status_code == 200
    assert tts(client, text="Back in business.").status_code == 200


def test_idle_worker_crash_keeps_the_pod_healthy(make_client):
    client = make_client(num_workers=2)
    wait_healthy(client)
    pid = client.get("/stats", headers=auth()).json()["workers"]["workers"][0]["pid"]
    os.kill(pid, signal.SIGKILL)
    wait_for(lambda: client.get("/stats", headers=auth()).json()["workers"]["respawns"] == 1)
    assert client.get("/health").status_code == 200
    wait_for(lambda: client.get("/health").json()["workers"]["ready"] == 2)


def test_auth(make_client):
    client = make_client()
    assert client.get("/health").status_code in (200, 503)
    assert client.get("/v1/voices").status_code == 401
    assert client.get("/stats", headers=auth("wrong")).status_code == 401
    assert client.post("/v1/tts", json={"text": "hi"}).status_code == 401


def test_validation_errors(make_client):
    client = make_client(max_text_chars=50)
    wait_healthy(client)
    cases = [
        ({"text": ""}, 400),
        ({"text": "x" * 51}, 413),
        ({"text": "hi", "voice": "nobody"}, 404),
        ({"text": "hi", "voice": "US_James", "style": "deep"}, 400),
        ({"text": "hi", "sample_rate": 16000}, 400),
        ({"text": "hi", "output_format": "mp3"}, 400),
        ({"text": "hi", "language": "xx"}, 400),
        ({"voice": "US_James"}, 400),
    ]
    for body, status in cases:
        response = client.post("/v1/tts", json=body, headers=auth())
        assert response.status_code == status, (body, response.text)
        assert isinstance(response.json()["detail"], str)


def test_voice_endpoints(make_client):
    client = make_client()
    wait_healthy(client)
    voices = {v["id"]: v for v in client.get("/v1/voices", headers=auth()).json()["voices"]}
    assert voices["US_James"]["kind"] == "builtin"
    assert voices["lingguang"] == {"id": "lingguang", "kind": "clone", "language": "zh",
                                   "prompt_seconds": 3.29,
                                   "created_at": "2026-10-26T00:00:00+00:00"}

    def upload(audio: bytes, **form):
        return client.post("/v1/voices/clone", headers=auth(),
                           files={"audio": ("ref.wav", audio, "audio/wav")},
                           data={"prompt_text": "Hello, this is my voice.", **form})

    created = upload(wav_bytes(2.5, channels=2), language="en", id="my_voice")
    assert created.status_code == 200, created.text
    assert created.json()["id"] == "my_voice" and created.json()["prompt_seconds"] == 2.5
    assert upload(wav_bytes(2.5), id="my_voice").status_code == 409
    assert upload(wav_bytes(0.4)).status_code == 400
    assert upload(b"\0" * (10 * 2**20 + 1)).status_code == 413
    generated = upload(wav_bytes(1.5))
    assert generated.json()["id"].startswith("clone_")

    response = tts(client, text="Speaking with my clone.", voice="my_voice")
    assert response.status_code == 200, response.text

    assert client.delete("/v1/voices/my_voice", headers=auth()).json() == {"ok": True, "id": "my_voice"}
    assert client.delete("/v1/voices/my_voice", headers=auth()).status_code == 404
    assert client.delete("/v1/voices/US_James", headers=auth()).status_code == 405
    assert tts(client, text="Gone.", voice="my_voice").status_code == 404
