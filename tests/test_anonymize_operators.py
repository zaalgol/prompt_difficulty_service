"""Unit tests for the consistency machinery: the custom operator and the vault.

These exercise coherence and distinctness directly, without the spaCy model, so
they are fast and deterministic-ish (Faker output varies, but the invariants —
same-in/same-out, all-distinct, fake != original — must always hold).
"""
import threading

import pytest

pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

fakeredis = pytest.importorskip("fakeredis")

from app.presidio_service.operators import (  # noqa: E402
    FAKER_BY_TYPE,
    ConsistentFakerAnonymizer,
    _fake_value,
)
from app.presidio_service.service import SessionVault  # noqa: E402


def _vault(**kwargs):
    """A SessionVault backed by an isolated in-process fakeredis."""
    return SessionVault(fakeredis.FakeStrictRedis(decode_responses=True), **kwargs)


def operate(text, mapping, entity_type="PERSON"):
    op = ConsistentFakerAnonymizer()
    return op.operate(text, {"entity_type": entity_type, "entity_mapping": mapping})


# ── coherence ────────────────────────────────────────────────────────────────

def test_same_value_same_fake():
    m = {}
    assert operate("John Smith", m) == operate("John Smith", m)


def test_case_and_whitespace_variants_collapse():
    m = {}
    base = operate("John Smith", m)
    assert operate("john smith", m) == base
    assert operate("  John   Smith  ", m) == base
    assert operate("JOHN SMITH", m) == base


def test_mapping_accumulates_one_entry_per_distinct_value():
    m = {}
    operate("John Smith", m)
    operate("john  smith", m)   # same normalized key
    operate("Jane Doe", m)
    assert len(m["PERSON"]) == 2


# ── distinctness (the "losing context" failure) ──────────────────────────────

def test_distinct_values_get_distinct_fakes():
    m = {}
    originals = [f"Person Number {i}" for i in range(60)]
    fakes = [operate(o, m) for o in originals]
    assert len(set(fakes)) == len(fakes), "fake values collided"


def test_fake_never_equals_original():
    m = {}
    for i in range(60):
        original = f"Unique Name {i}"
        assert operate(original, m).casefold() != original.casefold()


def test_unknown_entity_type_is_distinct_per_instance():
    m = {}
    a = operate("v1", m, entity_type="MYSTERY")
    b = operate("v2", m, entity_type="MYSTERY")
    c = operate("v3", m, entity_type="MYSTERY")
    assert a.startswith("<MYSTERY>")
    assert len({a, b, c}) == 3


# ── isolation between entity types ───────────────────────────────────────────

def test_same_string_different_types_do_not_share_fake():
    m = {}
    as_person = operate("Washington", m, entity_type="PERSON")
    as_location = operate("Washington", m, entity_type="LOCATION")
    # Independent mappings; the PERSON fake should not leak into LOCATION.
    assert "PERSON" in m and "LOCATION" in m
    assert m["PERSON"]["washington"] == as_person
    assert m["LOCATION"]["washington"] == as_location


# ── operator contract ────────────────────────────────────────────────────────

def test_validate_requires_entity_mapping():
    op = ConsistentFakerAnonymizer()
    with pytest.raises(ValueError):
        op.validate({"entity_type": "PERSON"})


def test_validate_rejects_non_dict_mapping():
    op = ConsistentFakerAnonymizer()
    with pytest.raises(ValueError):
        op.validate({"entity_mapping": ["not", "a", "dict"]})


def test_operator_identity():
    op = ConsistentFakerAnonymizer()
    assert op.operator_name() == "consistent_faker"


def test_faker_map_covers_common_types():
    for etype in ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                  "US_SSN", "IP_ADDRESS", "URL", "LOCATION", "IBAN_CODE", "DATE_TIME"]:
        assert etype in FAKER_BY_TYPE
        assert isinstance(_fake_value(etype), str) and _fake_value(etype)


# ── SessionVault (Redis-backed: load / save / delete) ────────────────────────

def test_vault_persists_mapping_across_load_save():
    v = _vault()
    mapping = v.load("s1")
    assert mapping == {}  # nothing stored yet
    mapping.setdefault("PERSON", {})["john smith"] = "Fake Name"
    v.save("s1", mapping)
    # A later request reloads the same mapping (a fresh dict with equal contents).
    reloaded = v.load("s1")
    assert reloaded == {"PERSON": {"john smith": "Fake Name"}}
    assert reloaded is not mapping


def test_vault_different_sessions_are_isolated():
    v = _vault()
    a = v.load("a")
    a.setdefault("PERSON", {})["x"] = "A"
    v.save("a", a)
    # A different session id never sees session "a"'s mapping.
    assert v.load("b") == {}


def test_vault_no_session_is_ephemeral():
    v = _vault()
    first = v.load(None)
    first.setdefault("PERSON", {})["john smith"] = "X"
    v.save(None, first)  # ephemeral save is a no-op
    second = v.load(None)
    # A brand-new mapping each time: nothing retained across calls.
    assert second == {}
    assert second is not first


def test_vault_delete_removes_session():
    v = _vault()
    m = v.load("s1")
    m.setdefault("PERSON", {})["x"] = "Y"
    v.save("s1", m)
    v.delete("s1")
    assert v.load("s1") == {}


def test_vault_refreshes_ttl_on_access():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    v = SessionVault(client, ttl_seconds=100)
    m = v.load("s1")
    m.setdefault("PERSON", {})["x"] = "Y"
    v.save("s1", m)
    assert 0 < client.ttl("anon:vault:s1") <= 100
    v.load("s1")  # access refreshes the TTL
    assert 0 < client.ttl("anon:vault:s1") <= 100


def test_vault_is_thread_safe():
    v = _vault()
    errors = []

    def worker(sid):
        try:
            for _ in range(50):
                m = v.load(sid)
                m.setdefault("PERSON", {})[str(threading.get_ident())] = "x"
                v.save(sid, m)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"sess-{i % 3}",)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
