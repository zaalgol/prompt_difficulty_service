import app.main as main
from app.main import _CLASSIFIER_WARMUP_PROMPT, _warmup_classifier


class FakePipeline:
    def __init__(self, warmed=True):
        self.warmed = warmed
        self.prompts = []

    def warm_inference(self, prompt):
        self.prompts.append(prompt)
        return self.warmed


def test_local_startup_uses_public_pipeline_warmup_api():
    pipeline = FakePipeline()

    _warmup_classifier({"pipeline": pipeline})

    assert pipeline.prompts == [_CLASSIFIER_WARMUP_PROMPT]


def test_non_local_pipeline_can_decline_dummy_inference():
    pipeline = FakePipeline(warmed=False)

    _warmup_classifier({"pipeline": pipeline})

    assert pipeline.prompts == [_CLASSIFIER_WARMUP_PROMPT]


def test_missing_artifact_and_pipeline_without_warmup_are_noops():
    _warmup_classifier(None)
    _warmup_classifier({"pipeline": object()})


def test_classifier_warmup_failure_does_not_abort_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "RUN_CLASSIFY_DUMMY_ON_STARTUP", True)
    monkeypatch.setattr(main, "RUN_ANONYMIZE_DUMMY_ON_STARTUP", True)
    monkeypatch.setattr(
        main,
        "_warmup_classifier",
        lambda artifact: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        main, "_warmup_anonymizer", lambda service: calls.append("anonymize")
    )

    main._run_startup_dummy_requests(
        {"pipeline": FakePipeline()}, object()
    )

    assert calls == ["anonymize"]
