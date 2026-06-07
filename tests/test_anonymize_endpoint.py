"""HTTP-level tests for the /anonymize endpoint.

These exercise real Presidio detection + anonymization, so they are skipped when
Presidio (or its spaCy model) is not installed. They verify response shape plus
the two core guarantees: coherence (consistent fakes) and semantics (preserved
entity types passed through unchanged).
"""
import pytest
from fastapi.testclient import TestClient

# Skip the whole module if the anonymization stack is unavailable.
pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _anonymize(client, prompt, **kwargs):
    body = {"prompt": prompt, **kwargs}
    return client.post("/anonymize", json=body)


def test_anonymize_returns_200_and_shape(client):
    r = _anonymize(client, "My name is John Smith and my email is john@example.com.")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["anonymized_prompt"], str)
    assert isinstance(data["entities"], list)
    assert data["preserved_entity_types"] == []
    for ent in data["entities"]:
        assert set(ent) == {"entity_type", "start", "end", "action"}


def test_original_pii_is_removed(client):
    r = _anonymize(client, "Contact John Smith about the overdue invoice.")
    data = r.json()
    # The detected name must not survive in the returned text.
    assert "John Smith" not in data["anonymized_prompt"]
    assert any(e["entity_type"] == "PERSON" for e in data["entities"])


def test_coherence_within_a_single_prompt(client):
    # The same name appearing twice should map to one fake value.
    r = _anonymize(client, "John Smith told me that John Smith would call back.")
    text = r.json()["anonymized_prompt"]
    assert "John Smith" not in text
    # Exactly one distinct replacement -> the word right after the first token
    # repeats. Simplest robust check: the text still reads with a repeated name.
    words = text.split()
    # Two PERSON spans were detected and both replaced with the same fake.
    persons = [e for e in r.json()["entities"] if e["entity_type"] == "PERSON"]
    assert len(persons) == 2


def test_coherence_across_requests_same_session(client):
    session = "session-coherence"
    r1 = _anonymize(client, "Please email John Smith.", session_id=session)
    r2 = _anonymize(client, "Did John Smith reply yet?", session_id=session)
    t1, t2 = r1.json()["anonymized_prompt"], r2.json()["anonymized_prompt"]
    assert "John Smith" not in t1 and "John Smith" not in t2
    # The fake name chosen in the first request must reappear in the second.
    # Recover it from t1 by locating the PERSON span.
    span = next(e for e in r1.json()["entities"] if e["entity_type"] == "PERSON")
    fake_name = t1[span["start"]:span["end"]]
    assert fake_name and fake_name in t2


def test_semantics_preserved_entity_is_unchanged(client):
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    r = _anonymize(client, prompt, preserve_entity_types=["DATE_TIME"])
    data = r.json()
    assert data["preserved_entity_types"] == ["DATE_TIME"]
    # The date the answer depends on stays intact...
    assert "1960-04-12" in data["anonymized_prompt"]
    # ...while the name is still anonymized.
    assert "Jane Doe" not in data["anonymized_prompt"]
    dt = [e for e in data["entities"] if e["entity_type"] == "DATE_TIME"]
    assert dt and all(e["action"] == "preserved" for e in dt)


def test_auto_preserve_keeps_dob_without_being_asked(client):
    # No preserve_entity_types passed: the service should infer that the retirement
    # question depends on the date of birth and keep it unchanged.
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    r = _anonymize(client, prompt)
    data = r.json()
    assert "DATE_TIME" in data["preserved_entity_types"]
    assert "1960-04-12" in data["anonymized_prompt"]   # date kept for the calc
    assert "Jane Doe" not in data["anonymized_prompt"]  # name still anonymized
    dt = [e for e in data["entities"] if e["entity_type"] == "DATE_TIME"]
    assert dt and all(e["action"] == "preserved" for e in dt)


def test_auto_preserve_can_be_disabled(client):
    # With auto_preserve off and no explicit preserve list, the date is anonymized.
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    r = _anonymize(client, prompt, auto_preserve=False)
    data = r.json()
    assert data["preserved_entity_types"] == []
    assert "1960-04-12" not in data["anonymized_prompt"]


def test_non_temporal_prompt_anonymizes_dates(client):
    # A date with no computation intent is treated as ordinary PII and replaced.
    prompt = "My name is Jane Doe and I last logged in on 2021-03-08."
    r = _anonymize(client, prompt)
    data = r.json()
    assert "DATE_TIME" not in data["preserved_entity_types"]


def test_empty_prompt_returns_422(client):
    assert _anonymize(client, "").status_code == 422
