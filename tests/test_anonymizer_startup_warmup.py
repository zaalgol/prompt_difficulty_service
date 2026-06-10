import pytest

import app.main as main
from app.main import (
    _ANONYMIZER_WARMUP_PROMPT,
    _run_startup_dummy_requests,
    _warmup_anonymizer,
)


class RecordingAnonymizer:
    def __init__(self):
        self.calls = []

    def anonymize(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "anonymized_prompt": "Contact Fake Person.",
            "session_id": None,
            "entities": [],
            "preserved_entity_types": [],
        }


def test_anonymizer_startup_runs_complete_ephemeral_request():
    service = RecordingAnonymizer()

    _warmup_anonymizer(service)

    assert service.calls == [
        {
            "prompt": _ANONYMIZER_WARMUP_PROMPT,
            "session_id": None,
            "preserve_entity_types": [],
            "auto_preserve": False,
        }
    ]


def test_anonymizer_warmup_prompt_exercises_pii_and_sensitive_data():
    assert "startup@example.com" in _ANONYMIZER_WARMUP_PROMPT
    assert "password" in _ANONYMIZER_WARMUP_PROMPT


@pytest.mark.parametrize(
    ("run_classify", "run_anonymize", "expected"),
    [
        (True, True, ["classify", "anonymize"]),
        (True, False, ["classify"]),
        (False, True, ["anonymize"]),
        (False, False, []),
    ],
)
def test_startup_flags_independently_control_dummy_requests(
    monkeypatch, run_classify, run_anonymize, expected
):
    calls = []
    monkeypatch.setattr(main, "RUN_CLASSIFY_DUMMY_ON_STARTUP", run_classify)
    monkeypatch.setattr(main, "RUN_ANONYMIZE_DUMMY_ON_STARTUP", run_anonymize)
    monkeypatch.setattr(main, "_warmup_classifier", lambda artifact: calls.append("classify"))
    monkeypatch.setattr(main, "_warmup_anonymizer", lambda service: calls.append("anonymize"))

    _run_startup_dummy_requests({"pipeline": object()}, RecordingAnonymizer())

    assert calls == expected


def test_startup_dummy_flags_default_to_true():
    assert main.RUN_CLASSIFY_DUMMY_ON_STARTUP is True
    assert main.RUN_ANONYMIZE_DUMMY_ON_STARTUP is True
