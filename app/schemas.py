from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    # Upper bound matches EmbeddingVectorizer.MAX_CHARS so the model classifies
    # the same text the caller sent (the embedding path truncates beyond this),
    # and to bound memory/CPU per request.
    prompt: str = Field(..., min_length=1, max_length=20_000)
    metadata: Optional[Dict[str, Any]] = None


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    method: str
    reason: str
    features: Dict[str, Any]


class LabelDatasetResponse(BaseModel):
    total_prompts: int
    label_counts: Dict[str, int]
    output_path: str


class TrainRequest(BaseModel):
    labeled_json_path: str
    model_output_path: Optional[str] = None


class TrainResponse(BaseModel):
    model_path: str
    total_examples: int
    label_counts: Dict[str, int]
    validation_accuracy: Optional[float]
    validation_false_cheap_rate: Optional[float]
