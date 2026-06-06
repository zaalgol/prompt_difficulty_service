"""
HTTP-level tests for the /classify endpoint.

These tests only verify the response structure — not label accuracy.
The service may use a trained model or the rule-based fallback depending
on what is configured in service_config.json and available on disk.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import LABEL_CHEAP_OK, LABEL_ESCALATE

VALID_LABELS = {LABEL_CHEAP_OK, LABEL_ESCALATE}
VALID_METHODS = {"trained_model", "rule_based_fallback"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_classify_returns_200(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    assert response.status_code == 200


def test_classify_response_has_required_fields(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "method" in data
    assert "reason" in data
    assert "features" in data


def test_classify_label_is_valid(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert data["label"] in VALID_LABELS


def test_classify_confidence_is_float_in_range(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_classify_method_is_valid(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert data["method"] in VALID_METHODS


def test_classify_model_version_is_string(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0


def test_classify_features_is_dict(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert isinstance(data["features"], dict)


def test_classify_reason_is_string(client):
    response = client.post("/classify", json={"prompt": "What is 2+2?"})
    data = response.json()
    assert isinstance(data["reason"], str)
    assert len(data["reason"]) > 0


def test_classify_empty_prompt_returns_422(client):
    response = client.post("/classify", json={"prompt": ""})
    assert response.status_code == 422


def test_classify_missing_prompt_returns_422(client):
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_health_returns_model_info(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_path" in data
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
