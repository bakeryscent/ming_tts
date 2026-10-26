"""The aligner process behind the HTTP layer, with ALIGNER=fake (no torch).

FAKE_ALIGN_S / FAKE_ALIGN_FAIL make the spawned aligner slow or broken on
purpose, so timeouts, errors and crashes can be observed. Every failure
mode must still answer 200 with estimated timestamps.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import ming_dir_or_none
from tests.helpers import ROOT, TOKEN, auth, wait_for
from tests.test_pool import tts, wait_healthy

TEXT = "The garden will remain open until sunset."
WORDS = ["The", "garden", "will", "remain", "open", "until", "sunset."]


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clients: list[TestClient] = []

    def _make(*, align_s: float = 0.0, align_fail: bool = False, **overrides) -> TestClient:
        monkeypatch.setenv("FAKE_LOAD_S", "0.1")
        monkeypatch.setenv("FAKE_SYNTH_S", "0.05")
        monkeypatch.setenv("FAKE_ALIGN_S", str(align_s))
        monkeypatch.setenv("FAKE_ALIGN_FAIL", "1" if align_fail else "0")
        settings = Settings(
            api_token=TOKEN, voices_dir=tmp_path / "voices", bundled_voices_dir=ROOT / "voices",
            ming_dir=ming_dir_or_none() or tmp_path / "no-ming", fake_engine=True,
            **{"num_workers": 1, "aligner": "fake", **overrides})
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


def aligner_stats(client: TestClient) -> dict:
    return client.get("/stats", headers=auth()).json()["aligner"]


def wait_aligner_ready(client: TestClient) -> dict:
    wait_healthy(client)
    return wait_for(lambda: (s := aligner_stats(client))["state"] == "idle" and s)


def test_aligned_timestamps(make_client):
    client = make_client()
    wait_aligner_ready(client)
    body = tts(client, text=TEXT).json()
    assert body["words_source"] == "aligned"
    assert [w["text"] for w in body["words"]] == WORDS
    starts = [w["start"] for w in body["words"]]
    assert starts == sorted(starts) and starts[0] == 0.0
    # The fake backend spreads the fragment evenly over its seven words.
    assert body["words"][-1]["end"] == pytest.approx(body["duration_seconds"], abs=0.002)
    lengths = {round(w["end"] - w["start"], 2) for w in body["words"]}
    assert len(lengths) == 1
    assert body["inference_status"]["align_ms"] >= 0
    stats = aligner_stats(client)
    assert stats["backend"] == "fake" and stats["jobs_done"] == 1 and stats["fallbacks"] == 0


def test_no_timestamps_means_no_alignment(make_client):
    client = make_client()
    wait_aligner_ready(client)
    body = tts(client, text=TEXT, return_timestamps=False).json()
    assert body["words"] == [] and body["words_source"] is None
    assert body["inference_status"]["align_ms"] is None
    assert aligner_stats(client)["jobs_done"] == 0


def test_aligner_off_keeps_the_estimate(make_client):
    client = make_client(aligner="off")
    wait_healthy(client)
    body = tts(client, text=TEXT).json()
    assert body["words_source"] == "estimated"
    assert body["inference_status"]["align_ms"] is None
    assert client.get("/stats", headers=auth()).json()["aligner"] == {"backend": "off"}


def test_requests_fall_back_while_the_aligner_is_loading(make_client, monkeypatch):
    client = make_client(align_s=0.0)
    wait_healthy(client)
    # The pool may be ready before the aligner: never wait for it.
    body = tts(client, text=TEXT).json()
    assert body["words_source"] in ("aligned", "estimated")
    assert [w["text"] for w in body["words"]] == WORDS


def test_alignment_timeout_falls_back_to_the_estimate(make_client):
    client = make_client(align_s=1.0, aligner_timeout_s=0.2)
    wait_aligner_ready(client)
    body = tts(client, text=TEXT).json()
    assert body["words_source"] == "estimated"
    assert [w["text"] for w in body["words"]] == WORDS
    assert 150 <= body["inference_status"]["align_ms"] < 900
    assert aligner_stats(client)["fallbacks"] == 1
    # The late result is dropped and the aligner is free again.
    wait_for(lambda: aligner_stats(client)["state"] == "idle"
             and aligner_stats(client)["jobs_done"] == 1)
    assert tts(client, text="Still fine.").status_code == 200


def test_alignment_error_falls_back_to_the_estimate(make_client):
    client = make_client(align_fail=True)
    wait_aligner_ready(client)
    body = tts(client, text=TEXT).json()
    assert body["words_source"] == "estimated"
    assert [w["text"] for w in body["words"]] == WORDS
    assert aligner_stats(client)["jobs_done"] == 1


def test_crashed_aligner_is_respawned_and_requests_keep_flowing(make_client):
    client = make_client()
    stats = wait_aligner_ready(client)
    os.kill(stats["pid"], signal.SIGKILL)
    wait_for(lambda: aligner_stats(client)["restarts"] == 1)
    response = tts(client, text=TEXT)
    assert response.status_code == 200
    assert response.json()["words_source"] in ("aligned", "estimated")
    revived = wait_for(lambda: (s := aligner_stats(client))["state"] == "idle" and s)
    assert revived["pid"] != stats["pid"]
    assert tts(client, text=TEXT).json()["words_source"] == "aligned"
    assert client.get("/health").status_code == 200


def test_aligner_crash_during_a_job_falls_back(make_client):
    client = make_client(align_s=2.0, aligner_timeout_s=5.0)
    wait_aligner_ready(client)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(tts, client, text=TEXT)
        busy = wait_for(lambda: (s := aligner_stats(client))["state"] == "busy" and s)
        os.kill(busy["pid"], signal.SIGKILL)
        response = pending.result(timeout=10)
    assert response.status_code == 200
    assert response.json()["words_source"] == "estimated"
    assert response.json()["inference_status"]["align_ms"] < 2000


def test_settings_from_env_validate_aligner(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "t")
    monkeypatch.setenv("ALIGNER", "MMS_FA")
    assert Settings.from_env().aligner == "mms_fa"
    monkeypatch.setenv("ALIGNER", "whisper")
    with pytest.raises(RuntimeError):
        Settings.from_env()
    monkeypatch.delenv("ALIGNER")
    assert Settings.from_env().aligner == "off"
