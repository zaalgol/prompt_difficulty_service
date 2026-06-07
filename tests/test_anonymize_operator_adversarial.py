"""Adversarial tests for the pseudonymization operator.

These tests avoid NLP detection entirely. They exercise the part that must act
like a bijection within a session: equal originals map to equal fakes, distinct
originals do not collapse, and no fallback may reveal the original value.
"""
import ipaddress
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import urlparse

import pytest

pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

import app.presidio_service.operators as operators
from app.presidio_service.operators import ConsistentFakerAnonymizer


def _operate(text, mapping, entity_type="PERSON"):
    return ConsistentFakerAnonymizer().operate(
        text,
        {"entity_type": entity_type, "entity_mapping": mapping},
    )


@pytest.mark.parametrize(
    "variant",
    [
        "john smith",
        "JOHN SMITH",
        "  John Smith  ",
        "John\tSmith",
        "John\nSmith",
        "John\u00a0Smith",
    ],
)
def test_whitespace_and_case_variants_share_one_mapping(variant):
    mapping = {}
    expected = _operate("John Smith", mapping)

    assert _operate(variant, mapping) == expected
    assert len(mapping["PERSON"]) == 1


def test_canonically_equivalent_unicode_names_share_one_mapping():
    mapping = {}
    composed = "Jos\u00e9 Garc\u00eda"
    decomposed = unicodedata.normalize("NFD", composed)

    assert composed != decomposed
    assert _operate(composed, mapping) == _operate(decomposed, mapping)
    assert len(mapping["PERSON"]) == 1


def test_large_mapping_remains_bijective():
    mapping = {}
    originals = [f"Sensitive Person {i:04d}" for i in range(500)]
    first_pass = [_operate(value, mapping) for value in originals]
    second_pass = [_operate(value, mapping) for value in reversed(originals)]

    assert len(set(first_pass)) == len(originals)
    assert second_pass == list(reversed(first_pass))
    assert len(mapping["PERSON"]) == len(originals)


def test_collision_exhaustion_never_returns_or_contains_original(monkeypatch):
    mapping = {}
    monkeypatch.setattr(operators, "_fake_value", lambda _entity_type: "John Smith")

    replacement = _operate("John Smith", mapping)

    assert "john smith" not in replacement.casefold()
    assert replacement.casefold() != "john smith"


def test_collision_exhaustion_still_produces_distinct_replacements(monkeypatch):
    mapping = {}
    monkeypatch.setattr(operators, "_fake_value", lambda _entity_type: "constant")

    replacements = [_operate(f"original-{i}", mapping) for i in range(20)]

    assert len(set(replacements)) == len(replacements)


def test_same_value_is_atomic_under_concurrent_requests(monkeypatch):
    """Two first sightings of one value must not receive different fakes."""
    mapping = {}
    counter_lock = threading.Lock()
    counter = 0

    def slow_unique_fake(_entity_type):
        nonlocal counter
        time.sleep(0.02)
        with counter_lock:
            counter += 1
            return f"Fake Person {counter}"

    monkeypatch.setattr(operators, "_fake_value", slow_unique_fake)
    start = threading.Barrier(12)

    def worker():
        start.wait()
        return _operate("John Smith", mapping)

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _i: worker(), range(12)))

    assert len(set(results)) == 1
    assert mapping["PERSON"]["john smith"] == results[0]


def test_many_concurrent_distinct_values_do_not_collapse(monkeypatch):
    mapping = {}
    counter_lock = threading.Lock()
    counter = 0

    def fake(_entity_type):
        nonlocal counter
        time.sleep(0.001)
        with counter_lock:
            counter += 1
            return f"Generated {counter}"

    monkeypatch.setattr(operators, "_fake_value", fake)
    originals = [f"Person {i}" for i in range(100)]

    with ThreadPoolExecutor(max_workers=20) as pool:
        replacements = list(pool.map(lambda value: _operate(value, mapping), originals))

    assert len(set(replacements)) == len(originals)
    assert len(mapping["PERSON"]) == len(originals)


@pytest.mark.parametrize(
    "entity_type, validator",
    [
        (
            "EMAIL_ADDRESS",
            lambda value: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value)),
        ),
        ("IP_ADDRESS", lambda value: bool(ipaddress.ip_address(value))),
        (
            "URL",
            lambda value: urlparse(value).scheme in {"http", "https"}
            and bool(urlparse(value).netloc),
        ),
        ("DATE_TIME", lambda value: bool(date.fromisoformat(value))),
        (
            "CREDIT_CARD",
            lambda value: 12 <= len(re.sub(r"\D", "", value)) <= 19,
        ),
    ],
)
def test_generated_values_preserve_machine_readable_shape(entity_type, validator):
    mapping = {}

    for index in range(20):
        replacement = _operate(f"original-{index}", mapping, entity_type)
        assert validator(replacement), (entity_type, replacement)


def test_mapping_for_one_type_does_not_mutate_another_type():
    mapping = {}
    person = _operate("Washington", mapping, "PERSON")
    before = dict(mapping["PERSON"])

    location = _operate("Washington", mapping, "LOCATION")

    assert mapping["PERSON"] == before
    assert mapping["PERSON"]["washington"] == person
    assert mapping["LOCATION"]["washington"] == location
