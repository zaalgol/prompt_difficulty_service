import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_CONFIG_PATH = PROJECT_ROOT / "service_config.json"


def _read_service_config() -> dict:
    """Best-effort read of service_config.json; never raises at import time."""
    try:
        if SERVICE_CONFIG_PATH.exists():
            with SERVICE_CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


_SERVICE_CONFIG = _read_service_config()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Logging — written to both the terminal and a rotating file under logs/.
# Level precedence: LOG_LEVEL env var > "log_level" in service_config.json > INFO.
# The env var stays the highest-priority override for per-deployment tuning.
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PATH = LOGS_DIR / "service.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", _SERVICE_CONFIG.get("log_level", "INFO")).upper()

DEFAULT_MODEL_PATH = MODELS_DIR / "prompt_classifier.joblib"
DEFAULT_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_embeddings.joblib"
DEFAULT_LGBM_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm.joblib"
DEFAULT_LGBM_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings.joblib"
DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings_tuned.joblib"
DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings_optuna.joblib"
DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_ensemble_embeddings.joblib"
DEFAULT_EMBEDDING_CACHE_PATH = DATA_DIR / "embeddings_cache.pkl"


def resolve_model_path(cfg: dict | None = None) -> Path:
    """Resolve the active model artifact path from service_config.json.

    Shared by the API (app.main) and the CLI (scripts/classify_prompt.py) so both
    select the same artifact. An empty/missing "model_path" falls back to the
    key-free TF-IDF baseline (DEFAULT_MODEL_PATH); if that file is absent, the
    downstream loader degrades to rule-based classification. Pass an already-read
    config dict as `cfg`, or omit it to read service_config.json fresh.
    """
    config = _read_service_config() if cfg is None else cfg
    raw = config.get("model_path")
    if not raw:
        return DEFAULT_MODEL_PATH
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path

# Embedding model used when training embedding-based variants. Override with
# "embedding_model" in service_config.json.
EMBEDDING_MODEL = _SERVICE_CONFIG.get("embedding_model", "text-embedding-3-small")

LABEL_CHEAP_OK = "cheap_ok"
LABEL_ESCALATE = "escalate"

MODEL_VERSION = "mvp-v1"

# Conservative default: if a cheap_ok prediction's confidence is below this
# threshold, route to escalate instead. Override with "min_cheap_confidence"
# in service_config.json.
MIN_CHEAP_CONFIDENCE = float(_SERVICE_CONFIG.get("min_cheap_confidence", 0.80))

# Presidio PII anonymization (/anonymize endpoint).
# - presidio_nlp_model: spaCy model the analyzer loads for English NER. The large
#   model (en_core_web_lg) is Presidio's default and the most accurate; en_core_web_sm
#   is lighter. Whatever is set here must be downloaded (`spacy download <model>`).
# - presidio_score_threshold: minimum detection confidence to act on an entity.
PRESIDIO_NLP_MODEL = _SERVICE_CONFIG.get("presidio_nlp_model", "en_core_web_lg")
PRESIDIO_SCORE_THRESHOLD = float(_SERVICE_CONFIG.get("presidio_score_threshold", 0.5))

# Where the /anonymize per-session vault (original->fake mappings) is stored:
#   "memory" — in-process dict (default; single-process, lost on restart)
#   "redis"  — Redis, so coherence survives across processes/restarts
# Set via the VAULT_BACKEND env var or "vault_backend" in service_config.json.
VAULT_BACKEND = os.environ.get(
    "VAULT_BACKEND", _SERVICE_CONFIG.get("vault_backend", "memory")
).strip().lower()

# In-memory backend: bound the number of retained sessions (LRU) so a stream of
# distinct caller session ids cannot grow process memory without limit.
PRESIDIO_MAX_SESSIONS = int(_SERVICE_CONFIG.get("presidio_max_sessions", 1000))

# Redis backend: connection string (REDIS_URL env var, else service_config.json,
# else a local default) and a short socket timeout so a missing backend fails fast
# (closed) instead of hanging the request.
REDIS_URL = os.environ.get("REDIS_URL", _SERVICE_CONFIG.get("redis_url", "redis://localhost:6379/0"))
REDIS_SOCKET_TIMEOUT = float(_SERVICE_CONFIG.get("redis_socket_timeout", 0.5))
# Redis backend: a session's mappings expire after this many seconds of inactivity
# (the TTL is refreshed on every request that touches the session), bounding memory.
PRESIDIO_SESSION_TTL_SECONDS = int(_SERVICE_CONFIG.get("presidio_session_ttl_seconds", 3600))

# Both backends: max original->fake mappings kept per entity type per session
# (FIFO eviction, enforced by the operator).
PRESIDIO_MAX_ENTRIES_PER_TYPE = int(_SERVICE_CONFIG.get("presidio_max_entries_per_type", 5000))

# Load the (heavy) spaCy/Presidio engines at startup instead of on first request.
# Off by default so a classify-only deployment starts fast; turn on where the
# service must be "ready" only once anonymization can actually run.
PRESIDIO_WARM_ON_STARTUP = bool(_SERVICE_CONFIG.get("presidio_warm_on_startup", False))
