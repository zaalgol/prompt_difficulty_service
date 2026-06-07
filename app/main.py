import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

from app.config import DEFAULT_MODEL_PATH, MODEL_VERSION, PROJECT_ROOT, SERVICE_CONFIG_PATH
from app.logging_config import get_logger
from app.modeling import classify_from_artifact, load_model, train_model
from app.presidio_service import AnonymizerService
from app.schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    ClassifyRequest,
    ClassifyResponse,
    TrainRequest,
    TrainResponse,
)

logger = get_logger(__name__)


def _load_config() -> Dict[str, Any]:
    if SERVICE_CONFIG_PATH.exists():
        with SERVICE_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_model_path(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("model_path")
    if not raw:
        return DEFAULT_MODEL_PATH
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_config()
    model_path = _resolve_model_path(cfg)
    artifact = load_model(model_path)

    app.state.model_path = model_path
    app.state.artifact = artifact
    # Lightweight to construct; the spaCy/Presidio engines load lazily on the
    # first /anonymize request so startup stays fast.
    app.state.anonymizer_service = AnonymizerService()

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
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, body: ClassifyRequest) -> ClassifyResponse:
    logger.debug("Classify request: %d chars", len(body.prompt))
    result = classify_from_artifact(body.prompt, request.app.state.artifact)
    return ClassifyResponse(**result)


@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize(request: Request, body: AnonymizeRequest) -> AnonymizeResponse:
    """Detect PII in a prompt and replace it with realistic, consistent fakes.

    Same value -> same fake across prompts sharing a session_id (coherence).
    Entity types in preserve_entity_types are kept unchanged so the answer stays
    correct (semantics). The prompt text is never logged.
    """
    service: AnonymizerService = request.app.state.anonymizer_service
    try:
        result = service.anonymize(
            prompt=body.prompt,
            session_id=body.session_id,
            preserve_entity_types=body.preserve_entity_types,
            auto_preserve=body.auto_preserve,
        )
    except ImportError as exc:
        # presidio / spaCy not installed (or the NLP model is missing).
        logger.exception("Anonymization dependencies unavailable")
        raise HTTPException(
            status_code=503, detail="Anonymization unavailable."
        ) from exc
    except Exception as exc:
        logger.exception("Anonymization failed")
        raise HTTPException(status_code=500, detail="Anonymization failed.") from exc

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
