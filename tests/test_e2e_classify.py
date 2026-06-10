"""
End-to-end tests: starts a real uvicorn server, sends actual HTTP requests.

Run with:
    .venv/Scripts/pytest tests/test_e2e_classify.py -v
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8765"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_LABELS = {"cheap_ok", "escalate"}
VALID_METHODS = {"trained_model", "rule_based_fallback"}


@pytest.fixture(scope="session", autouse=True)
def server():
    """Start uvicorn in a subprocess and wait until it is accepting requests."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8765",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll /health until the server is up. The configured trained model (ensemble
    # embeddings) is loaded during startup, which can take tens of seconds, so the
    # deadline is generous.
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.3)
    else:
        proc.terminate()
        stdout = proc.stdout.read().decode(errors="replace")
        stderr = proc.stderr.read().decode(errors="replace")
        pytest.fail(f"Server did not start in time.\nstdout: {stdout}\nstderr: {stderr}")

    yield

    proc.terminate()
    proc.wait(timeout=5)


# ── helpers ──────────────────────────────────────────────────────────────────

def classify(prompt: str) -> httpx.Response:
    return httpx.post(f"{BASE_URL}/classify", json={"prompt": prompt}, timeout=30)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health_ok():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_path" in data
    assert "model_loaded" in data


def test_classify_returns_200():
    assert classify("What is 2+2?").status_code == 200


def test_classify_response_has_required_fields():
    data = classify("What is 2+2?").json()
    for field in (
        "label",
        "confidence",
        "model_version",
        "method",
        "reason",
        "features",
        "elapsed_time_ms",
    ):
        assert field in data, f"Missing field: {field}"


def test_classify_label_is_valid():
    data = classify("What is 2+2?").json()
    assert data["label"] in VALID_LABELS


def test_classify_confidence_is_float_in_range():
    data = classify("What is 2+2?").json()
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_classify_method_is_valid():
    data = classify("What is 2+2?").json()
    assert data["method"] in VALID_METHODS


def test_classify_features_is_dict():
    data = classify("What is 2+2?").json()
    assert isinstance(data["features"], dict)


def test_classify_empty_prompt_returns_422():
    r = classify("")
    assert r.status_code == 422


def test_classify_missing_prompt_returns_422():
    r = httpx.post(f"{BASE_URL}/classify", json={}, timeout=10)
    assert r.status_code == 422
