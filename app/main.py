import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.config import DEFAULT_MODEL_PATH, MODEL_VERSION, PROJECT_ROOT, SERVICE_CONFIG_PATH
from app.dataset import label_dataset
from app.modeling import classify_from_artifact, load_model, train_model
from app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    LabelDatasetResponse,
    TrainRequest,
    TrainResponse,
)


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
        print(f"[startup] Loaded model '{version}' from {model_path}")
    else:
        print(f"[startup] No model at {model_path} — using rule-based fallback")

    yield


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
    return {
        "status": "ok",
        "service": "prompt-difficulty-classifier",
        "model_path": str(model_path),
        "model_loaded": artifact is not None,
        "model_version": artifact.get("model_version") if artifact else None,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, body: ClassifyRequest) -> ClassifyResponse:
    result = classify_from_artifact(body.prompt, request.app.state.artifact)
    return ClassifyResponse(**result)


@app.post("/label-dataset", response_model=LabelDatasetResponse)
async def label_dataset_endpoint(
    file: UploadFile = File(...),
) -> LabelDatasetResponse:
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a .json file.")

    try:
        content = await file.read()

        with NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(content)
            input_path = Path(tmp.name)

        output_path = Path("data") / f"{Path(file.filename).stem}_labeled_binary.json"

        total, counts = label_dataset(input_path, output_path)

        return LabelDatasetResponse(
            total_prompts=total,
            label_counts=counts,
            output_path=str(output_path),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    try:
        model_output_path = request.model_output_path or str(DEFAULT_MODEL_PATH)
        result = train_model(
            labeled_json_path=request.labeled_json_path,
            model_output_path=model_output_path,
        )
        return TrainResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
