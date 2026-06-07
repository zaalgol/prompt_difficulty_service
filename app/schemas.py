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


class AnonymizeRequest(BaseModel):
    # Upper bound matches ClassifyRequest so the same prompt can flow through both
    # endpoints, and to bound CPU per request.
    prompt: str = Field(..., min_length=1, max_length=20_000)
    # Scopes the consistency mapping: the same value maps to the same fake value
    # across prompts that share a session_id. Omit for one-shot anonymization.
    session_id: Optional[str] = None
    # Entity types to keep unchanged because the answer depends on the real value
    # (e.g. ["DATE_TIME"] for a date of birth in a retirement question).
    preserve_entity_types: List[str] = Field(default_factory=list)
    # When true, the service also auto-detects values the answer depends on (e.g.
    # a date of birth when the prompt asks about retirement) and keeps them. Set
    # false to anonymize strictly by preserve_entity_types only.
    auto_preserve: bool = True


class AnonymizedEntity(BaseModel):
    entity_type: str
    start: int
    end: int
    action: str  # "anonymized" or "preserved"


class AnonymizeResponse(BaseModel):
    # Original PII values are deliberately omitted; they stay server-side.
    anonymized_prompt: str
    session_id: Optional[str]
    entities: List[AnonymizedEntity]
    preserved_entity_types: List[str]


class TrainRequest(BaseModel):
    labeled_json_path: str
    model_output_path: Optional[str] = None


class TrainResponse(BaseModel):
    model_path: str
    total_examples: int
    label_counts: Dict[str, int]
    validation_accuracy: Optional[float]
    validation_false_cheap_rate: Optional[float]
