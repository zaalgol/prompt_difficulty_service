"""Anonymizer service: Presidio engines + per-session consistency vault."""
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from app.config import (
    PRESIDIO_MAX_SESSIONS,
    PRESIDIO_NLP_MODEL,
    PRESIDIO_SCORE_THRESHOLD,
)
from app.logging_config import get_logger
from app.presidio_service.operators import ConsistentFakerAnonymizer
from app.presidio_service.semantics import infer_required_entity_types

logger = get_logger(__name__)

# Sentinel session used when the caller does not supply a session_id. Coherence
# still holds within a single prompt, but nothing is retained across requests.
_NO_SESSION = "__ephemeral__"


class SessionVault:
    """Holds the original -> fake mappings per session, keyed by entity type.

    Structure: {session_id: {entity_type: {original_value: fake_value}}}.

    The original PII lives here, in process memory only. It is never returned to
    callers or written to logs. Access is guarded by a lock because FastAPI runs
    sync endpoints in a threadpool, so several requests may touch the vault at once.

    The number of retained sessions is bounded (LRU): the least-recently-used
    session is evicted once max_sessions is exceeded, so a stream of distinct
    caller-controlled session ids cannot exhaust memory. Sessions are still
    process-local and cleared on restart (an MVP constraint, documented).
    """

    def __init__(self, max_sessions: int = PRESIDIO_MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._max_sessions = max(1, max_sessions)
        self._sessions: "OrderedDict[str, Dict[str, Dict[str, str]]]" = OrderedDict()

    def mapping_for(self, session_id: Optional[str]) -> Dict[str, Dict[str, str]]:
        key = session_id or _NO_SESSION
        with self._lock:
            if key == _NO_SESSION:
                # Fresh mapping each call: no cross-request coherence wanted.
                return {}
            if key in self._sessions:
                self._sessions.move_to_end(key)  # mark recently used
                return self._sessions[key]
            mapping: Dict[str, Dict[str, str]] = {}
            self._sessions[key] = mapping
            if len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)  # evict least-recently-used
            return mapping


class AnonymizerService:
    """Detects PII with presidio-analyzer and replaces it with presidio-anonymizer.

    The analyzer/anonymizer engines load a spaCy model and are therefore built
    lazily on first use, so importing this module (and starting the app) stays
    cheap and does not require Presidio to be installed unless /anonymize is hit.
    """

    def __init__(self) -> None:
        self._vault = SessionVault()
        self._lock = threading.Lock()
        self._analyzer = None
        self._anonymizer = None

    @property
    def engines_loaded(self) -> bool:
        """Whether the spaCy/Presidio engines are loaded (no side effects)."""
        return self._analyzer is not None and self._anonymizer is not None

    def status(self) -> Dict[str, Any]:
        """Readiness detail for /health (does not trigger a load)."""
        return {"engines_loaded": self.engines_loaded, "nlp_model": PRESIDIO_NLP_MODEL}

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

        mapping = self._vault.mapping_for(session_id)
        operators: Dict[str, OperatorConfig] = {
            entity_type: OperatorConfig("keep") for entity_type in preserve
        }
        operators["DEFAULT"] = OperatorConfig(
            "consistent_faker", {"entity_mapping": mapping}
        )

        result = self._anonymizer.anonymize(
            text=prompt, analyzer_results=analyzer_results, operators=operators
        )

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
