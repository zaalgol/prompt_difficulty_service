"""Recognizer for developer credentials and secrets commonly found in prompts."""
import re
from dataclasses import dataclass
from typing import List, Optional

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult


@dataclass(frozen=True)
class _SecretPattern:
    name: str
    entity_type: str
    regex: re.Pattern
    group: Optional[str] = None
    score: float = 0.95


_FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL


def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, _FLAGS)


_SECRET_LABEL = (
    r"(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|apikey|secret(?:_access)?_key|client_secret|"
    r"password|passwd|pwd|token|access_token|refresh_token|access_key_id|"
    r"accountkey|private_token|docker_auth)"
)

_CONNECTION_URI = (
    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|"
    r"amqp|amqps|mssql)://[^:\s/@]+:(?P<secret>[^@\s/]+)(?=@)"
)

_PROTECTED_CONTEXT_PATTERNS = [
    _compile(
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|"
        r"amqp|amqps|mssql)://[^\s\"'`]+"
    ),
    _compile(
        rf"[\"']?{_SECRET_LABEL}[\"']?\s*[:=]\s*"
        r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|\$\{[^}\r\n]+\}|"
        r"<[^>\r\n]+>|[^\s,;#}\r\n]+)"
    ),
    _compile(r"\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[^\s,;]+"),
    _compile(
        r"\bCookie\s*:\s*[^\r\n]*(?:session|sessionid|session_token|auth|"
        r"access_token|refresh_token)\s*=\s*[^\s;]+"
    ),
]


# More specific patterns come first. Later, contextual patterns cover generic
# assignments while replacing only the value, not the key or surrounding syntax.
_PATTERNS = [
    _SecretPattern(
        "private_key",
        "PRIVATE_KEY",
        _compile(
            r"-----BEGIN (?P<kind>(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY)-----"
            r".*?"
            r"-----END (?P=kind)-----"
        ),
    ),
    _SecretPattern(
        "aws_access_key",
        "CLOUD_ACCESS_KEY",
        _compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    _SecretPattern(
        "openai_api_key",
        "API_KEY",
        _compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    ),
    _SecretPattern(
        "anthropic_api_key",
        "API_KEY",
        _compile(r"(?<![A-Za-z0-9])sk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{20,}"),
    ),
    _SecretPattern(
        "github_token",
        "API_KEY",
        _compile(
            r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,255}"
            r"|github_pat_[A-Za-z0-9_]{20,255})"
        ),
    ),
    _SecretPattern(
        "gitlab_token",
        "API_KEY",
        _compile(r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}"),
    ),
    _SecretPattern(
        "slack_token",
        "AUTH_TOKEN",
        _compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    ),
    _SecretPattern(
        "stripe_key",
        "API_KEY",
        _compile(r"(?<![A-Za-z0-9])(?:sk|rk)_(?:test|live)_[A-Za-z0-9]{16,}"),
    ),
    _SecretPattern(
        "google_api_key",
        "API_KEY",
        _compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"),
    ),
    _SecretPattern(
        "jwt",
        "AUTH_TOKEN",
        _compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}"
            r"\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"
        ),
    ),
    _SecretPattern(
        "authorization_header",
        "AUTH_TOKEN",
        _compile(
            r"\bAuthorization\s*:\s*(?:Bearer|Basic)\s+"
            r"(?P<secret>[A-Za-z0-9._~+/=-]{8,})"
        ),
        group="secret",
    ),
    _SecretPattern(
        "bearer_token",
        "AUTH_TOKEN",
        _compile(r"\bBearer\s+(?P<secret>[A-Za-z0-9._~+/=-]{12,})"),
        group="secret",
    ),
    _SecretPattern(
        "credentialed_connection_uri",
        "PASSWORD",
        _compile(_CONNECTION_URI),
        group="secret",
    ),
    _SecretPattern(
        "cookie_secret",
        "AUTH_TOKEN",
        _compile(
            r"\b(?:session|sessionid|session_token|auth|access_token|refresh_token)"
            r"\s*=\s*(?P<secret>[A-Za-z0-9._~+/=-]{8,})"
        ),
        group="secret",
    ),
    _SecretPattern(
        "labeled_secret",
        "SECRET",
        _compile(
            rf"[\"']?{_SECRET_LABEL}[\"']?"
            r"\s*[:=]\s*[\"']?"
            r"(?P<secret>[A-Za-z0-9_./+=:@!-]{8,})"
        ),
        group="secret",
        score=0.9,
    ),
]


def protected_context_spans(text: str) -> List[tuple[int, int]]:
    """Regions where generic NER must not override technical syntax."""
    return [
        match.span()
        for pattern in _PROTECTED_CONTEXT_PATTERNS
        for match in pattern.finditer(text)
    ]


def is_sensitive_result(result: RecognizerResult) -> bool:
    metadata = result.recognition_metadata or {}
    return metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY) == "SensitiveDataRecognizer"


class SensitiveDataRecognizer(EntityRecognizer):
    """Detect high-confidence credentials while preserving surrounding syntax."""

    def __init__(self) -> None:
        super().__init__(
            supported_entities=sorted({item.entity_type for item in _PATTERNS}),
            name="SensitiveDataRecognizer",
            supported_language="en",
            version="1.0.0",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results: List[RecognizerResult] = []
        claimed: List[tuple[int, int]] = []

        for item in _PATTERNS:
            if entities and item.entity_type not in entities:
                continue
            for match in item.regex.finditer(text):
                start, end = match.span(item.group) if item.group else match.span()
                if item.name == "labeled_secret":
                    candidate = text[start:end]
                    if end < len(text) and text[end] == "(":
                        continue
                    if candidate.casefold() in {"os.getenv", "env.get", "process.env"}:
                        continue
                if start == end or any(start < other_end and end > other_start for other_start, other_end in claimed):
                    continue

                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    original_score=item.score,
                    pattern_name=item.name,
                    pattern=item.regex.pattern,
                    textual_explanation="Matched a high-confidence developer secret pattern.",
                    regex_flags=item.regex.flags,
                )
                results.append(
                    RecognizerResult(
                        entity_type=item.entity_type,
                        start=start,
                        end=end,
                        score=item.score,
                        analysis_explanation=explanation,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
                claimed.append((start, end))

        return results
