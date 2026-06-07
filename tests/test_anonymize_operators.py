"""Unit tests for the consistency machinery: the custom operator and the vault.

These exercise coherence and distinctness directly, without the spaCy model, so
they are fast and deterministic-ish (Faker output varies, but the invariants —
same-in/same-out, all-distinct, fake != original — must always hold).
"""
import threading

import pytest

pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

from app.presidio_service.operators import (  # noqa: E402
    FAKER_BY_TYPE,
    ConsistentFakerAnonymizer,
    _fake_value,
)
from app.presidio_service.service import SessionVault  # noqa: E402


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


# ── SessionVault ─────────────────────────────────────────────────────────────

def test_vault_same_session_returns_same_mapping_object():
    v = SessionVault()
    m1 = v.mapping_for("s1")
    m1.setdefault("PERSON", {})["john smith"] = "Fake Name"
    m2 = v.mapping_for("s1")
    assert m2 is m1
    assert m2["PERSON"]["john smith"] == "Fake Name"


def test_vault_different_sessions_are_isolated():
    v = SessionVault()
    a = v.mapping_for("a")
    b = v.mapping_for("b")
    assert a is not b


def test_vault_no_session_is_ephemeral():
    v = SessionVault()
    first = v.mapping_for(None)
    first.setdefault("PERSON", {})["john smith"] = "X"
    second = v.mapping_for(None)
    # A brand-new mapping each time: nothing retained across calls.
    assert second == {}
    assert second is not first


def test_vault_is_thread_safe():
    v = SessionVault()
    errors = []

    def worker(sid):
        try:
            for _ in range(200):
                m = v.mapping_for(sid)
                m.setdefault("PERSON", {})[str(threading.get_ident())] = "x"
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"sess-{i % 3}",)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
