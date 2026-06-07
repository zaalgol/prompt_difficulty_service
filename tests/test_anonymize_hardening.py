"""Tests for the privacy/robustness hardening added from the review:

- replacements use reserved, non-routable values (cannot target real systems);
- the session vault is bounded (per-session Redis TTL, capped entries per type);
- request validation bounds session_id and preserve_entity_types;
- /health reports anonymizer readiness without forcing a load.

None of these need the spaCy model — they exercise the operator, the vault, the
schema, and the endpoint contract directly.
"""
import re

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")
fakeredis = pytest.importorskip("fakeredis")

from app.presidio_service.operators import ConsistentFakerAnonymizer  # noqa: E402
from app.presidio_service.service import SessionVault  # noqa: E402


def _vault(**kwargs):
    return SessionVault(fakeredis.FakeStrictRedis(decode_responses=True), **kwargs)


def _operate(text, mapping, entity_type):
    op = ConsistentFakerAnonymizer()
    return op.operate(text, {"entity_type": entity_type, "entity_mapping": mapping})


# ── safe reserved replacements ───────────────────────────────────────────────

def test_ip_replacement_is_in_documentation_range():
    # RFC 5737 TEST-NET-1 (192.0.2.0/24) — never routable.
    for original in ["8.8.8.8", "203.0.113.7", "10.0.0.1"]:
        fake = _operate(original, {}, "IP_ADDRESS")
        assert re.fullmatch(r"192\.0\.2\.\d{1,3}", fake), fake


def test_email_replacement_uses_reserved_domain():
    fake = _operate("alice@realbank.com", {}, "EMAIL_ADDRESS")
    assert fake.endswith("@example.com")  # RFC 2606 reserved


def test_url_replacement_uses_reserved_domain():
    fake = _operate("https://internal.corp/secret", {}, "URL")
    assert fake.startswith("https://example.com/")


def test_phone_replacement_uses_fictional_range():
    fake = _operate("+1 415 555 9988", {}, "PHONE_NUMBER")
    assert re.fullmatch(r"\(555\) 555-01\d{2}", fake), fake


def test_safe_replacements_are_still_coherent_and_distinct():
    m = {}
    a1 = _operate("8.8.8.8", m, "IP_ADDRESS")
    a2 = _operate("8.8.8.8", m, "IP_ADDRESS")
    b = _operate("1.1.1.1", m, "IP_ADDRESS")
    assert a1 == a2          # coherence preserved
    assert a1 != b           # distinctness preserved


# ── bounded session vault (Redis TTL) ────────────────────────────────────────

def test_vault_sets_ttl_so_sessions_expire():
    # Memory is bounded by a per-session TTL instead of an LRU session cap: every
    # stored session carries an expiry, so abandoned sessions cannot accumulate.
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    v = SessionVault(client, ttl_seconds=60)
    m = v.load("a")
    m.setdefault("PERSON", {})["x"] = "y"
    v.save("a", m)
    assert 0 < client.ttl("anon:vault:a") <= 60


def test_vault_expired_session_comes_back_empty():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    v = SessionVault(client, ttl_seconds=60)
    m = v.load("a")
    m.setdefault("PERSON", {})["x"] = "y"
    v.save("a", m)
    client.delete("anon:vault:a")  # simulate TTL expiry
    assert v.load("a") == {}


def test_per_type_entry_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(
        "app.presidio_service.operators.PRESIDIO_MAX_ENTRIES_PER_TYPE", 3
    )
    mapping = {}
    for i in range(10):
        _operate(f"Person {i}", mapping, "PERSON")
    assert len(mapping["PERSON"]) <= 3


# ── request validation ───────────────────────────────────────────────────────

class _Recording:
    def __init__(self):
        self.calls = []

    def anonymize(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "anonymized_prompt": "ok",
            "session_id": kwargs.get("session_id"),
            "entities": [],
            "preserved_entity_types": [],
        }

    def status(self):
        return {"engines_loaded": False, "nlp_model": "test"}


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("body", [
    {"prompt": "x", "session_id": "s" * 257},                 # session_id too long
    {"prompt": "x", "preserve_entity_types": ["date_time"]},  # not UPPER_SNAKE
    {"prompt": "x", "preserve_entity_types": ["A B"]},        # space not allowed
    {"prompt": "x", "preserve_entity_types": [""]},           # empty type
    {"prompt": "x", "preserve_entity_types": ["T"] * 33},     # too many types
])
def test_invalid_policy_inputs_are_rejected_before_service(client, body):
    fake = _Recording()
    client.app.state.anonymizer_service = fake
    response = client.post("/anonymize", json=body)
    assert response.status_code == 422
    assert fake.calls == []


def test_valid_policy_inputs_are_accepted(client):
    fake = _Recording()
    client.app.state.anonymizer_service = fake
    response = client.post("/anonymize", json={
        "prompt": "x",
        "session_id": "abc-123",
        "preserve_entity_types": ["DATE_TIME", "EMAIL_ADDRESS"],
    })
    assert response.status_code == 200
    assert fake.calls[0]["preserve_entity_types"] == ["DATE_TIME", "EMAIL_ADDRESS"]


def test_redis_backend_unavailable_fails_closed_503(client):
    from app.presidio_service import AnonymizerBackendUnavailable

    class _Down:
        def anonymize(self, **kwargs):
            raise AnonymizerBackendUnavailable()

        def status(self):
            return {"engines_loaded": False, "nlp_model": "test", "redis_connected": False}

    client.app.state.anonymizer_service = _Down()
    response = client.post("/anonymize", json={"prompt": "Contact John Smith.", "session_id": "s"})
    assert response.status_code == 503
    # The error must not leak prompt text.
    assert "John Smith" not in response.text


# ── health readiness ─────────────────────────────────────────────────────────

def test_health_reports_anonymizer_status(client):
    data = client.get("/health").json()
    assert "anonymizer" in data
    assert set(data["anonymizer"]) == {"engines_loaded", "nlp_model", "redis_connected"}
    # Lazy by default: engines are not loaded just because the process is up.
    assert data["anonymizer"]["engines_loaded"] is False
    # Vault backend reachability is reported (fakeredis is up under tests).
    assert data["anonymizer"]["redis_connected"] is True
