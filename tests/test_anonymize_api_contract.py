"""HTTP contract tests for /anonymize with a deterministic fake service.

Real Presidio behavior is covered elsewhere. These tests isolate validation,
argument forwarding, response filtering, and safe error handling.
"""
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("presidio_anonymizer")

from app.main import app


class RecordingAnonymizer:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or {
            "anonymized_prompt": "Hello Fake Person",
            "session_id": None,
            "entities": [
                {
                    "entity_type": "PERSON",
                    "start": 6,
                    "end": 17,
                    "action": "anonymized",
                }
            ],
            "preserved_entity_types": [],
        }
        self.error = error

    def anonymize(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    # These are endpoint contract tests with an injected fake service. Do not
    # run application lifespan: loading Nomic and Presidio for every
    # parameterized validation case would test startup repeatedly, not the API.
    test_client = TestClient(app)
    test_client.app.state.anonymizer_service = RecordingAnonymizer()
    yield test_client
    test_client.close()


def _install(client, service):
    client.app.state.anonymizer_service = service
    return service


def test_endpoint_forwards_every_policy_input_without_mutation(client):
    fake = _install(client, RecordingAnonymizer())

    response = client.post(
        "/anonymize",
        json={
            "prompt": "Contact John Smith on 2025-01-01.",
            "session_id": "session-123",
            "preserve_entity_types": ["DATE_TIME", "EMAIL_ADDRESS"],
            "auto_preserve": False,
        },
    )

    assert response.status_code == 200
    assert fake.calls == [
        {
            "prompt": "Contact John Smith on 2025-01-01.",
            "session_id": "session-123",
            "preserve_entity_types": ["DATE_TIME", "EMAIL_ADDRESS"],
            "auto_preserve": False,
        }
    ]


def test_endpoint_applies_documented_defaults(client):
    fake = _install(client, RecordingAnonymizer())

    response = client.post("/anonymize", json={"prompt": "Contact John Smith."})

    assert response.status_code == 200
    assert fake.calls[0]["session_id"] is None
    assert fake.calls[0]["preserve_entity_types"] == []
    assert fake.calls[0]["auto_preserve"] is True


def test_response_includes_non_negative_elapsed_time(client):
    _install(client, RecordingAnonymizer())

    data = client.post(
        "/anonymize", json={"prompt": "Contact John Smith."}
    ).json()

    assert isinstance(data["elapsed_time_ms"], float)
    assert data["elapsed_time_ms"] >= 0


def test_response_model_drops_unexpected_sensitive_fields(client):
    fake = RecordingAnonymizer(
        result={
            "anonymized_prompt": "Contact Fake Person.",
            "session_id": "s",
            "entities": [],
            "preserved_entity_types": [],
            "original_prompt": "Contact John Smith.",
            "entity_mapping": {"John Smith": "Fake Person"},
        }
    )
    _install(client, fake)

    data = client.post(
        "/anonymize",
        json={"prompt": "Contact John Smith.", "session_id": "s"},
    ).json()

    assert "original_prompt" not in data
    assert "entity_mapping" not in data
    assert "John Smith" not in repr(data)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": None},
        {"prompt": 123},
        {"prompt": "x", "preserve_entity_types": "DATE_TIME"},
        {"prompt": "x", "preserve_entity_types": [1]},
        {"prompt": "x", "auto_preserve": "not-a-boolean"},
    ],
)
def test_invalid_requests_are_rejected_before_service_call(client, body):
    fake = _install(client, RecordingAnonymizer())

    response = client.post("/anonymize", json=body)

    assert response.status_code == 422
    assert fake.calls == []


def test_prompt_at_size_limit_is_accepted(client):
    fake = _install(client, RecordingAnonymizer())
    prompt = "x" * 20_000

    response = client.post("/anonymize", json={"prompt": prompt})

    assert response.status_code == 200
    assert fake.calls[0]["prompt"] == prompt


def test_prompt_over_size_limit_is_rejected_before_service_call(client):
    fake = _install(client, RecordingAnonymizer())

    response = client.post("/anonymize", json={"prompt": "x" * 20_001})

    assert response.status_code == 422
    assert fake.calls == []


def test_missing_dependency_returns_sanitized_503(client):
    _install(client, RecordingAnonymizer(error=ImportError("secret install path")))

    response = client.post("/anonymize", json={"prompt": "Contact John Smith."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Anonymization unavailable."}
    assert "secret install path" not in response.text


def test_internal_failure_returns_sanitized_500(client):
    _install(client, RecordingAnonymizer(error=RuntimeError("John Smith leaked here")))

    response = client.post("/anonymize", json={"prompt": "Contact John Smith."})

    assert response.status_code == 500
    assert response.json() == {"detail": "Anonymization failed."}
    assert "John Smith leaked here" not in response.text
