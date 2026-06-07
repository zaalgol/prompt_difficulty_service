"""End-to-end tests for /anonymize: starts a real uvicorn server, sends real HTTP.

Run with:
    .venv/Scripts/pytest tests/test_e2e_anonymize.py -v

Skipped when the Presidio stack (or its spaCy model) is not installed, since the
server subprocess shares this venv and could not anonymize without it.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

from app.config import REDIS_URL  # noqa: E402
from app.presidio_service.service import _KEY_PREFIX  # noqa: E402

BASE_URL = "http://127.0.0.1:8766"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The server subprocess shares this venv but NOT the in-process fakeredis patch, so
# cross-session coherence here needs a real Redis at REDIS_URL. Ephemeral requests
# (no session_id) never touch Redis, so only the session-based tests require it.
_E2E_SESSION_PREFIX = "e2e-"


def _real_redis():
    """Return a connected redis client for REDIS_URL, or None if unreachable."""
    try:
        import redis

        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=0.5)
        client.ping()
        return client
    except Exception:
        return None


@pytest.fixture(scope="module")
def redis_or_skip():
    client = _real_redis()
    if client is None:
        pytest.skip(f"no Redis reachable at {REDIS_URL}; session-coherence tests need it")
    return client


@pytest.fixture(scope="module", autouse=True)
def _cleanup_e2e_sessions():
    """Delete this module's session keys afterward so reruns have no side effects."""
    yield
    client = _real_redis()
    if client is None:
        return
    keys = list(client.scan_iter(match=f"{_KEY_PREFIX}{_E2E_SESSION_PREFIX}*"))
    if keys:
        client.delete(*keys)


@pytest.fixture(scope="module", autouse=True)
def server():
    """Start uvicorn in a subprocess and wait until it is accepting requests."""
    # Route server output to a real file, not a PIPE. Presidio/spaCy log heavily on
    # model load; an undrained PIPE would fill its OS buffer and deadlock the server
    # mid-request. A file never blocks the writer, and we can read it on failure.
    log_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8766",
        ],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    def _server_log() -> str:
        log_file.seek(0)
        return log_file.read().decode(errors="replace")

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
        pytest.fail(f"Server did not start in time.\nserver log:\n{_server_log()}")

    yield

    proc.terminate()
    proc.wait(timeout=5)
    log_file.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def anonymize(prompt: str, **kwargs) -> httpx.Response:
    body = {"prompt": prompt, **kwargs}
    # The first request triggers a lazy spaCy/Presidio load; allow generous time.
    return httpx.post(f"{BASE_URL}/anonymize", json=body, timeout=60)


def person_span_text(resp: httpx.Response) -> str:
    """Return the substring the first PERSON entity was replaced with."""
    data = resp.json()
    span = next(e for e in data["entities"] if e["entity_type"] == "PERSON")
    return data["anonymized_prompt"][span["start"]:span["end"]]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_anonymize_returns_200_and_shape():
    r = anonymize("My name is John Smith and my email is john@example.com.")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["anonymized_prompt"], str)
    assert isinstance(data["entities"], list)
    assert data["preserved_entity_types"] == []
    for ent in data["entities"]:
        assert set(ent) == {"entity_type", "start", "end", "action"}


def test_original_pii_is_removed():
    r = anonymize("Contact John Smith about the overdue invoice.")
    data = r.json()
    assert "John Smith" not in data["anonymized_prompt"]
    assert any(e["entity_type"] == "PERSON" for e in data["entities"])


def test_coherence_within_a_single_prompt():
    r = anonymize("John Smith told me that John Smith would call back.")
    assert "John Smith" not in r.json()["anonymized_prompt"]
    persons = [e for e in r.json()["entities"] if e["entity_type"] == "PERSON"]
    assert len(persons) == 2


def test_coherence_across_requests_same_session(redis_or_skip):
    session = "e2e-session-coherence"
    r1 = anonymize("Please email John Smith.", session_id=session)
    r2 = anonymize("Did John Smith reply yet?", session_id=session)
    fake_name = person_span_text(r1)
    assert fake_name and "John Smith" not in r1.json()["anonymized_prompt"]
    # The fake chosen in the first request must reappear in the second.
    assert fake_name in r2.json()["anonymized_prompt"]


def test_different_sessions_are_isolated(redis_or_skip):
    r1 = anonymize("Please email John Smith.", session_id="e2e-sess-a")
    r2 = anonymize("Please email John Smith.", session_id="e2e-sess-b")
    # Both anonymize the name; mappings are per-session (values may differ).
    assert "John Smith" not in r1.json()["anonymized_prompt"]
    assert "John Smith" not in r2.json()["anonymized_prompt"]


def test_semantics_preserved_entity_is_unchanged():
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    r = anonymize(prompt, preserve_entity_types=["DATE_TIME"])
    data = r.json()
    assert data["preserved_entity_types"] == ["DATE_TIME"]
    assert "1960-04-12" in data["anonymized_prompt"]
    assert "Jane Doe" not in data["anonymized_prompt"]
    dt = [e for e in data["entities"] if e["entity_type"] == "DATE_TIME"]
    assert dt and all(e["action"] == "preserved" for e in dt)


def test_auto_preserve_keeps_dob_without_being_asked():
    # No preserve_entity_types: the service infers the retirement question depends
    # on the date of birth and keeps it unchanged.
    r = anonymize("I am Jane Doe, born 1960-04-12. When do I retire at 67?")
    data = r.json()
    assert "DATE_TIME" in data["preserved_entity_types"]
    assert "1960-04-12" in data["anonymized_prompt"]
    assert "Jane Doe" not in data["anonymized_prompt"]


def test_empty_prompt_returns_422():
    assert anonymize("").status_code == 422
