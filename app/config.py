import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_CONFIG_PATH = PROJECT_ROOT / "service_config.json"

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Logging — written to both the terminal and a rotating file under logs/.
# Level can be overridden with the LOG_LEVEL env var (e.g. DEBUG, INFO, WARNING).
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PATH = LOGS_DIR / "service.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

DEFAULT_MODEL_PATH = MODELS_DIR / "prompt_classifier.joblib"
DEFAULT_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_embeddings.joblib"
DEFAULT_LGBM_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm.joblib"
DEFAULT_LGBM_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings.joblib"
DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings_tuned.joblib"
DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH = MODELS_DIR / "prompt_classifier_lgbm_embeddings_optuna.joblib"
DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH = MODELS_DIR / "prompt_classifier_ensemble_embeddings.joblib"
DEFAULT_EMBEDDING_CACHE_PATH = DATA_DIR / "embeddings_cache.pkl"

EMBEDDING_MODEL = "text-embedding-3-small"

LABEL_CHEAP_OK = "cheap_ok"
LABEL_ESCALATE = "escalate"

MODEL_VERSION = "mvp-v1"

# Conservative default.
# If classifier confidence is lower than this threshold, route to escalate.
MIN_CHEAP_CONFIDENCE = 0.80
