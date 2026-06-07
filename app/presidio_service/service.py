"""Anonymizer service: Presidio engines + Redis-backed per-session vault."""
import json
import threading
from typing import Any, Dict, List, Optional

from app.config import (
    PRESIDIO_NLP_MODEL,
    PRESIDIO_SCORE_THRESHOLD,
    PRESIDIO_SESSION_TTL_SECONDS,
    REDIS_SOCKET_TIMEOUT,
    REDIS_URL,
)
from app.logging_config import get_logger
from app.presidio_service.operators import ConsistentFakerAnonymizer
from app.presidio_service.semantics import infer_required_entity_types
from app.presidio_service.sensitive_data import (
    SensitiveDataRecognizer,
    is_sensitive_result,
    protected_context_spans,
)

logger = get_logger(__name__)

# Imported at module top so `except RedisError` works without redis installed at
# import time; if redis-py is genuinely missing, _make_redis_client raises a clear
# ImportError on first use instead.
try:
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - redis is a declared dependency
    class RedisError(Exception):
        """Fallback when redis-py is not importable."""

# Redis key prefix for vault entries: anon:vault:{session_id} -> JSON blob of
# {entity_type: {normalized_original: fake_value}}.
_KEY_PREFIX = "anon:vault:"


class AnonymizerBackendUnavailable(RuntimeError):
    """Raised when the Redis vault backend cannot be reached, so /anonymize can
    fail closed (503) rather than silently losing cross-request coherence."""


def _make_redis_client() -> Any:
    """Build a Redis client from REDIS_URL. Imported lazily so importing this
    module does not require redis-py, and patched in tests to use fakeredis."""
    import redis

    return redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
    )


class SessionVault:
    """Stores the original -> fake mappings per session in Redis.

    Each session is one Redis key (anon:vault:{session_id}) holding a JSON blob of
    {entity_type: {normalized_original: fake_value}}. The blob is loaded into a
    plain dict at the start of a request and written back with a refreshed TTL at
    the end, which keeps the in-process operator a simple dict mutator while
    coherence survives across processes and restarts.

    The original PII lives in Redis only; it is never returned to callers or
    written to logs. A session expires after ttl_seconds of inactivity (refreshed
    on every access), bounding memory without an explicit session count cap.

    Concurrency note: concurrent requests for the *same* session load-modify-save
    independently, so two brand-new values created at the exact same time may not
    merge (last write wins). Values already in Redis stay coherent. Anonymize
    calls within one conversation are normally sequential, so this is acceptable
    for the MVP — and strictly better than the old per-process in-memory vault,
    which shared nothing across workers.
    """

    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: int = PRESIDIO_SESSION_TTL_SECONDS,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        self._client = client
        self._ttl = max(1, int(ttl_seconds))
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def ping(self) -> None:
        """Raise a RedisError if the backend is unreachable."""
        self._client.ping()

    def load(self, session_id: Optional[str]) -> Dict[str, Dict[str, str]]:
        """Return the session's nested mapping (empty for an ephemeral request)."""
        if not session_id:
            # No session id: no cross-request coherence wanted. The operator still
            # gets intra-prompt coherence from this fresh dict.
            return {}
        key = self._key(session_id)
        raw = self._client.get(key)
        if not raw:
            return {}
        # Refresh TTL on access so active sessions do not expire mid-conversation.
        self._client.expire(key, self._ttl)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            entity_type: dict(values)
            for entity_type, values in data.items()
            if isinstance(values, dict)
        }

    def save(self, session_id: Optional[str], mapping: Dict[str, Dict[str, str]]) -> None:
        """Persist the session's mapping with a refreshed TTL (no-op if ephemeral
        or empty). Rewrites the whole blob so operator-side evictions propagate."""
        if not session_id or not mapping:
            return
        self._client.set(self._key(session_id), json.dumps(mapping), ex=self._ttl)

    def delete(self, session_id: str) -> None:
        """Remove a session's mappings (used by tests for cleanup)."""
        self._client.delete(self._key(session_id))


class AnonymizerService:
    """Detects PII with presidio-analyzer and replaces it with presidio-anonymizer.

    The analyzer/anonymizer engines load a spaCy model and are therefore built
    lazily on first use, so importing this module (and starting the app) stays
    cheap and does not require Presidio to be installed unless /anonymize is hit.
    """

    def __init__(self, redis_client: Any = None) -> None:
        # The client is created eagerly but does not connect until first use, so a
        # classify-only deployment still starts without a running Redis.
        client = redis_client if redis_client is not None else _make_redis_client()
        self._vault = SessionVault(client)
        self._lock = threading.Lock()
        self._analyzer = None
        self._anonymizer = None

    @property
    def engines_loaded(self) -> bool:
        """Whether the spaCy/Presidio engines are loaded (no side effects)."""
        return self._analyzer is not None and self._anonymizer is not None

    def _redis_connected(self) -> bool:
        try:
            self._vault.ping()
            return True
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        """Readiness detail for /health (does not trigger an engine load)."""
        return {
            "engines_loaded": self.engines_loaded,
            "nlp_model": PRESIDIO_NLP_MODEL,
            # Cross-request coherence requires Redis; surface it so a deployment
            # can gate readiness on the vault backend being reachable.
            "redis_connected": self._redis_connected(),
        }

    def warmup(self) -> None:
        """Eagerly load the engines (e.g. at startup) so the first request is fast
        and readiness reflects that anonymization can actually run."""
        self._ensure_engines()

    def _ensure_engines(self) -> None:
        if self._analyzer is not None and self._anonymizer is not None:
            return
        with self._lock:
            if self._analyzer is not None and self._anonymizer is not None:
                return
            # Imported here (not at module top) so the rest of the app runs without
            # Presidio installed; the endpoint surfaces a clear 503 if it is missing.
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            logger.info("Loading Presidio engines (spaCy model '%s')", PRESIDIO_NLP_MODEL)
            nlp_engine = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": PRESIDIO_NLP_MODEL}],
                }
            ).create_engine()

            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            analyzer.registry.add_recognizer(SensitiveDataRecognizer())
            anonymizer = AnonymizerEngine()
            anonymizer.add_anonymizer(ConsistentFakerAnonymizer)

            self._analyzer = analyzer
            self._anonymizer = anonymizer
            logger.info("Presidio engines ready")

    def anonymize(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        preserve_entity_types: Optional[List[str]] = None,
        auto_preserve: bool = True,
    ) -> Dict[str, Any]:
        """Return the prompt with PII replaced, plus a description of each entity.

        preserve_entity_types lists entity types whose original value must be kept
        unchanged because the answer depends on it (e.g. DATE_TIME for a retirement
        question). Everything else is replaced with a consistent fake value.

        When auto_preserve is True (default), a keyword heuristic also infers types
        the answer depends on (e.g. it keeps DATE_TIME when the prompt asks a
        retirement/age/countdown question), and unions them with the explicit list.
        Auto-detection only ever adds preservation; it never anonymizes more.
        """
        from presidio_anonymizer.entities import OperatorConfig

        self._ensure_engines()

        preserve_set = set(preserve_entity_types or [])
        if auto_preserve:
            preserve_set |= set(infer_required_entity_types(prompt))
        preserve = sorted(preserve_set)

        analyzer_results = self._analyzer.analyze(
            text=prompt, language="en", score_threshold=PRESIDIO_SCORE_THRESHOLD
        )
        sensitive_results = [item for item in analyzer_results if is_sensitive_result(item)]
        protected_spans = protected_context_spans(prompt)

        def overlaps(start: int, end: int, spans) -> bool:
            return any(start < span_end and end > span_start for span_start, span_end in spans)

        # High-confidence credential matches take precedence over generic NER.
        # Also protect technical assignment/URI syntax from spaCy false positives.
        analyzer_results = [
            item
            for item in analyzer_results
            if is_sensitive_result(item)
            or (
                not overlaps(
                    item.start,
                    item.end,
                    [(secret.start, secret.end) for secret in sensitive_results],
                )
                and not overlaps(item.start, item.end, protected_spans)
                and not (
                    item.entity_type == "ORGANIZATION"
                    and prompt[item.start:item.end].strip(" =:").casefold() == "api"
                )
                and not (
                    item.entity_type == "DATE_TIME"
                    and prompt[item.start:item.end].casefold()
                    in {"monthly", "weekly", "daily", "yearly", "annually"}
                )
            )
        ]

        # Load the session's existing mappings from Redis. Fail closed (503) if the
        # backend is unreachable rather than silently dropping cross-request
        # coherence by starting from an empty mapping.
        try:
            mapping = self._vault.load(session_id)
        except RedisError as exc:
            logger.error("Anonymizer vault unavailable on load (%s)", type(exc).__name__)
            raise AnonymizerBackendUnavailable() from exc

        operators: Dict[str, OperatorConfig] = {
            entity_type: OperatorConfig("keep") for entity_type in preserve
        }
        operators["DEFAULT"] = OperatorConfig(
            "consistent_faker", {"entity_mapping": mapping}
        )

        result = self._anonymizer.anonymize(
            text=prompt, analyzer_results=analyzer_results, operators=operators
        )

        # Persist any new mappings the operator added (with a refreshed TTL). Also
        # fail closed here: returning a result we could not record would let a
        # later request mint a different fake for the same value.
        try:
            self._vault.save(session_id, mapping)
        except RedisError as exc:
            logger.error("Anonymizer vault unavailable on save (%s)", type(exc).__name__)
            raise AnonymizerBackendUnavailable() from exc

        entities = [
            {
                "entity_type": item.entity_type,
                "start": item.start,
                "end": item.end,
                "action": "preserved" if item.entity_type in preserve_set else "anonymized",
            }
            for item in result.items
        ]
        # result.items come back in reverse text order; present them left-to-right.
        entities.sort(key=lambda e: e["start"])

        logger.info(
            "Anonymized prompt (session=%s): %d entities, %d preserved",
            session_id or "-", len(entities), len(preserve_set & {e["entity_type"] for e in entities}),
        )

        return {
            "anonymized_prompt": result.text,
            "session_id": session_id,
            "entities": entities,
            "preserved_entity_types": preserve,
        }
