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

# Bound the in-memory session vault so a long-lived process (or an abusive caller
# pumping unique values/session ids) cannot grow memory without limit. Oldest
# sessions are evicted first; within a session each entity type keeps at most
# PRESIDIO_MAX_ENTRIES_PER_TYPE original->fake mappings (FIFO eviction).
PRESIDIO_MAX_SESSIONS = int(_SERVICE_CONFIG.get("presidio_max_sessions", 1000))
PRESIDIO_MAX_ENTRIES_PER_TYPE = int(_SERVICE_CONFIG.get("presidio_max_entries_per_type", 5000))

# Load the (heavy) spaCy/Presidio engines at startup instead of on first request.
# Off by default so a classify-only deployment starts fast; turn on where the
# service must be "ready" only once anonymization can actually run.
PRESIDIO_WARM_ON_STARTUP = bool(_SERVICE_CONFIG.get("presidio_warm_on_startup", False))
