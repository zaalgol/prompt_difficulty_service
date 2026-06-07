"""Heuristics that decide which PII must be kept for the answer to stay correct.

Some questions can only be answered from the *real* value. The classic case is a
date of birth together with a question like "when do I retire?": anonymizing the
date would make the model compute the wrong answer. When the prompt shows such
intent, the relevant entity types are preserved instead of replaced.

This is a deliberately conservative, keyword-based heuristic — not semantic
parsing. It only ever *adds* preservation; it never causes more data to be
anonymized. Callers can still preserve types explicitly, and can turn this off
per request (see AnonymizeRequest.auto_preserve).

Trade-off: a false positive keeps a real date that may not have been needed (a
privacy leak); a false negative anonymizes a date the answer depended on (a wrong
answer). The patterns below target clear date/age computation intent to keep false
positives low.
"""
import re
from typing import List

# Intent signals meaning the answer is computed from a date/age in the prompt:
# retirement, age, countdowns to/from a date, expiry/renewal, anniversaries, etc.
# Matched case-insensitively against the prompt.
_DATE_INTENT_PATTERNS = [
    r"\bretire", r"\bretirement\b",
    r"\b(?:when|what\s+(?:date|age)|how\s+long)\b.*?\bpension\b",
    r"\bpension\b.*?\b(?:when|what\s+(?:date|age)|start(?:s|ed|ing)?|eligible)\b",
    r"\bhow old\b",
    r"\bwhat age\b",
    r"\b(my|his|her|their|your|the)\s+age\b",
    r"\bwhen\s+(will|do|does|did|can|is|am|are|would|should)\b",
    r"\bhow\s+(long|many\s+years|many\s+days|many\s+months|many\s+weeks)\b",
    r"\byears?\s+(until|till|to|old|from\s+now|since|ago)\b",
    r"\banniversary\b",
    r"\b(?:when|how\s+long)\b.*?\bexpir(?:e|es|ed|ing|y|ation)\b",
    r"\bexpir(?:e|es|ed|ing|y|ation)\b.*?\bwhen\b",
    r"\b(?:expiry|expiration)\s+date\b",
    r"\brenew(?:al|als|ed|ing|s)?\b",          # renew(al/ed) — not "renewable"
    r"\bdue\s+date\b",
    r"\bdeadline\b",
    r"\bcountdown\b",
    r"\bdays?\s+(until|till|left|remaining)\b",
]

_DATE_INTENT_RE = re.compile("|".join(_DATE_INTENT_PATTERNS), re.IGNORECASE | re.DOTALL)

# Presidio entity types to keep unchanged when date/age intent is detected.
_DATE_ENTITY_TYPES = ["DATE_TIME"]


def infer_required_entity_types(prompt: str) -> List[str]:
    """Return entity types whose real value the answer depends on (may be empty)."""
    required: List[str] = []
    if _DATE_INTENT_RE.search(prompt):
        required.extend(_DATE_ENTITY_TYPES)
    return required
