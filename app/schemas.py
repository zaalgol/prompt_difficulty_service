from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
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
