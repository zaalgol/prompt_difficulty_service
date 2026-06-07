import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.config import DEFAULT_MODEL_PATH, MODEL_VERSION, PROJECT_ROOT, SERVICE_CONFIG_PATH
from app.dataset import label_dataset
from app.logging_config import get_logger
from app.modeling import classify_from_artifact, load_model, train_model
from app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    LabelDatasetResponse,
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


@app.post("/label-dataset", response_model=LabelDatasetResponse)
async def label_dataset_endpoint(
    file: UploadFile = File(...),
) -> LabelDatasetResponse:
    if not file.filename or not file.filename.endswith(".json"):
        logger.warning("Rejected upload with invalid filename: %r", file.filename)
        raise HTTPException(status_code=400, detail="Please upload a .json file.")

    input_path: Optional[Path] = None
    try:
        content = await file.read()

        with NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(content)
            input_path = Path(tmp.name)

        output_path = Path("data") / f"{Path(file.filename).stem}_labeled_binary.json"

        logger.info("Labeling dataset from upload '%s'", file.filename)
        total, counts = label_dataset(input_path, output_path)
        logger.info("Labeled %d prompts -> %s (counts=%s)", total, output_path, counts)

        return LabelDatasetResponse(
            total_prompts=total,
            label_counts=counts,
            output_path=str(output_path),
        )

    except Exception as exc:
        logger.exception("Failed to label dataset from upload '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Failed to label dataset.") from exc
    finally:
        # Always remove the temp upload, including on failure, so prompt history
        # is not left behind in the system temp directory.
        if input_path is not None:
            try:
                os.unlink(input_path)
            except OSError:
                logger.warning("Could not remove temp upload %s", input_path)


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
