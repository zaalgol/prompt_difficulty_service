import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

from app.config import (
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
    RUN_ANONYMIZE_DUMMY_ON_STARTUP,
    RUN_CLASSIFY_DUMMY_ON_STARTUP,
    SERVICE_CONFIG_PATH,
    resolve_model_path,
)
from app.logging_config import get_logger
from app.modeling import classify_from_artifact, load_model, train_model
from app.presidio_service import AnonymizerBackendUnavailable, AnonymizerService
from app.schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    ClassifyRequest,
    ClassifyResponse,
    TrainRequest,
    TrainResponse,
)

logger = get_logger(__name__)

_CLASSIFIER_WARMUP_PROMPT = (
    "Classify this harmless startup prompt to initialize local inference."
)
_ANONYMIZER_WARMUP_PROMPT = (
    "Contact Startup Example at startup@example.com using password "
    "'warmup-value-12345'."
)


def _load_config() -> Dict[str, Any]:
    if SERVICE_CONFIG_PATH.exists():
        with SERVICE_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _elapsed_time_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _warmup_classifier(artifact: Optional[Dict[str, Any]]) -> None:
    """Initialize local embedding and classifier inference before readiness.

    Loading a PyTorch model does not initialize every CPU kernel and thread pool.
    Run the same predict + predict_proba path used by /classify so that one-time
    work happens during startup. OpenAI artifacts are deliberately excluded to
    avoid making a paid network request for a synthetic prompt.
    """
    if not artifact:
        return

    pipeline = artifact.get("pipeline")
    if hasattr(pipeline, "warm_inference") and pipeline.warm_inference(
        _CLASSIFIER_WARMUP_PROMPT
    ):
        logger.info("Local classifier inference warmup complete")


def _warmup_anonymizer(service: AnonymizerService) -> None:
    """Run the complete analyzer + anonymizer path before readiness.

    No session id is supplied, so fake values generated for this synthetic
    prompt are ephemeral and never enter the memory or Redis session vault.
    """
    service.anonymize(
        prompt=_ANONYMIZER_WARMUP_PROMPT,
        session_id=None,
        preserve_entity_types=[],
        auto_preserve=False,
    )
    logger.info("Anonymizer inference warmup complete")


def _run_startup_dummy_requests(
    artifact: Optional[Dict[str, Any]],
    anonymizer_service: AnonymizerService,
) -> None:
    if RUN_CLASSIFY_DUMMY_ON_STARTUP:
        try:
            _warmup_classifier(artifact)
        except Exception as exc:
            # Request-time inference fails closed to escalate, so startup should
            # remain available under the same degraded behavior.
            logger.warning(
                "Classifier warmup failed (%s); requests will initialize lazily",
                type(exc).__name__,
            )

    if RUN_ANONYMIZE_DUMMY_ON_STARTUP:
        try:
            _warmup_anonymizer(anonymizer_service)
        except Exception:
            # Keep classification available when the optional anonymization
            # stack fails; /anonymize will surface the dependency error.
            logger.warning("Anonymizer warmup failed; engines will load on first use")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_config()
    model_path = resolve_model_path(cfg)
    artifact = load_model(model_path)

    app.state.model_path = model_path
    app.state.artifact = artifact
    anonymizer_service = AnonymizerService()
    _run_startup_dummy_requests(artifact, anonymizer_service)
    app.state.anonymizer_service = anonymizer_service

    if artifact:
        version = artifact.get("model_version", "unknown")
        logger.info("Loaded model '%s' from %s", version, model_path)
    else:
        logger.warning("No model at %s — using rule-based fallback", model_path)

    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="Prompt Difficulty Classification Service",
    version=MODEL_VERSION,
    description=(
        "MVP service for binary prompt difficulty classification. "
        "Labels: cheap_ok / escalate."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    model_path: Path = request.app.state.model_path
    artifact: Optional[dict] = request.app.state.artifact
    model_loaded = artifact is not None
    anonymizer: AnonymizerService = request.app.state.anonymizer_service
    return {
        # Liveness: the process is up. Readiness: the configured trained model
        # is loaded. When the model is missing the service still answers (rule
        # fallback) but readiness is false so deployments can detect the
        # degraded mode instead of silently trusting an unintended classifier.
        "status": "ok",
        "ready": model_loaded,
        "mode": "trained_model" if model_loaded else "rule_based_fallback",
        "service": "prompt-difficulty-classifier",
        "model_path": str(model_path),
        "model_loaded": model_loaded,
        "model_version": artifact.get("model_version") if model_loaded else None,
        # Anonymizer engines load lazily by default, so this reflects whether the
        # NLP model is loaded — not just that the process is up. A deployment that
        # requires anonymization should gate readiness on engines_loaded.
        "anonymizer": anonymizer.status(),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, body: ClassifyRequest) -> ClassifyResponse:
    started_at = time.perf_counter()
    logger.debug("Classify request: %d chars", len(body.prompt))
    result = classify_from_artifact(
        body.prompt, request.app.state.artifact, body.total_input_tokens,
    )
    result["elapsed_time_ms"] = _elapsed_time_ms(started_at)
    return ClassifyResponse(**result)


@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize(request: Request, body: AnonymizeRequest) -> AnonymizeResponse:
    """Detect PII in a prompt and replace it with realistic, consistent fakes.

    Same value -> same fake across prompts sharing a session_id (coherence).
    Entity types in preserve_entity_types are kept unchanged so the answer stays
    correct (semantics). The prompt text is never logged.
    """
    started_at = time.perf_counter()
    service: AnonymizerService = request.app.state.anonymizer_service
    try:
        result = service.anonymize(
            prompt=body.prompt,
            session_id=body.session_id,
            preserve_entity_types=body.preserve_entity_types,
            auto_preserve=body.auto_preserve,
        )
    except AnonymizerBackendUnavailable as exc:
        # Redis vault unreachable: fail closed so cross-request coherence is never
        # silently lost. No prompt text is in this exception.
        logger.error("Anonymization backend unavailable")
        raise HTTPException(
            status_code=503, detail="Anonymization backend unavailable."
        ) from exc
    except ImportError as exc:
        # presidio / spaCy not installed (or the NLP model is missing). Log only
        # the exception class — not the message/traceback, which can embed prompt
        # text or paths from deep in the dependency stack.
        logger.error("Anonymization unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Anonymization unavailable."
        ) from exc
    except Exception as exc:
        # Same reasoning: a dependency exception message could contain the prompt.
        logger.error("Anonymization failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Anonymization failed.") from exc

    result["elapsed_time_ms"] = _elapsed_time_ms(started_at)
    return AnonymizeResponse(**result)


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    try:
        model_output_path = request.model_output_path or str(DEFAULT_MODEL_PATH)
        logger.info(
            "Training requested: input=%s output=%s",
            request.labeled_json_path, model_output_path,
        )
        result = train_model(
            labeled_json_path=request.labeled_json_path,
            model_output_path=model_output_path,
        )
        return TrainResponse(**result)
    except Exception as exc:
        logger.exception("Training failed for input %s", request.labeled_json_path)
        raise HTTPException(status_code=500, detail="Training failed.") from exc
