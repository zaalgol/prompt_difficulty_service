"""Presidio-based PII anonymization service.

Detects sensitive information in a prompt and replaces it with realistic fake
values before the prompt is forwarded to an LLM. Two properties are guaranteed:

- Coherence: the same original value maps to the same fake value across prompts
  of the same session (see SessionVault).
- Semantics: entity types the caller marks as required for the answer are kept
  unchanged (Presidio's "keep" operator).
"""
from app.presidio_service.service import (
    AnonymizerBackendUnavailable,
    AnonymizerService,
)

__all__ = ["AnonymizerService", "AnonymizerBackendUnavailable"]
