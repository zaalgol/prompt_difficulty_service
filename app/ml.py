"""Pure Python ML utilities — no compiled extensions required."""
import json as _json
import math
import os
import pickle
import random
import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(text: str, lowercase: bool, ngram_range: Tuple[int, int]) -> List[str]:
    if lowercase:
        text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    result: List[str] = []
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            result.append(" ".join(tokens[i : i + n]))
    return result


# ---------------------------------------------------------------------------
# Token-count feature
# ---------------------------------------------------------------------------
# The classifier can use the prompt's `total_input_tokens` as one extra feature
# alongside the text features. Raw counts span 0..tens-of-thousands, which would
# dominate the normalized text features it is appended to, so we standardize
# log1p(count) to ~zero-mean/unit-variance. Tree models are scale-insensitive,
# but logistic regression is not, so we scale for all model variants uniformly.

class TokenScaler:
    """Standardizes log1p(token_count) so the single token-count feature sits on
    the same scale as the text features it is appended to."""

    def __init__(self) -> None:
        self.mean_ = 0.0
        self.std_ = 1.0

    @staticmethod
    def _log(t: Any) -> float:
        try:
            return math.log1p(max(0.0, float(t)))
        except (TypeError, ValueError):
            return 0.0

    def fit(self, tokens: List[Any]) -> "TokenScaler":
        vals = [self._log(t) for t in tokens if t is not None]
        n = len(vals)
        if not n:
            return self
        self.mean_ = sum(vals) / n
        var = sum((v - self.mean_) ** 2 for v in vals) / n
        self.std_ = math.sqrt(var) or 1.0
        return self

    def transform_one(self, t: Any) -> float:
        # An unknown count (None) maps to the standardized mean (0.0) so a caller
        # that omits the token count gets a neutral feature, not a fabricated one.
        if t is None:
            return 0.0
        return (self._log(t) - self.mean_) / self.std_


def _as_tokens(tokens: Optional[List[Any]], n: int) -> List[Any]:
    """Coerce a tokens argument aligned to n samples; None -> all-unknown."""
    if tokens is None:
        return [None] * n
    return list(tokens)


def _augment_dense_rows(
    rows: List[List[float]], tokens: Optional[List[Any]], scaler: "TokenScaler",
) -> List[List[float]]:
    """Append one standardized token-count feature to each dense feature row."""
    toks = _as_tokens(tokens, len(rows))
    return [list(r) + [scaler.transform_one(t)] for r, t in zip(rows, toks)]


def _augment_sparse_rows(
    rows: List[Dict[int, float]],
    tokens: Optional[List[Any]],
    scaler: "TokenScaler",
    token_index: int,
) -> List[Dict[int, float]]:
    """Add the standardized token-count feature at `token_index` to each row."""
    toks = _as_tokens(tokens, len(rows))
    out: List[Dict[int, float]] = []
    for r, t in zip(rows, toks):
        d = dict(r)
        d[token_index] = scaler.transform_one(t)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# TF-IDF vectoriser
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    def __init__(
        self,
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        min_df: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._n_docs: int = 0

    def _tokens(self, text: str) -> List[str]:
        return _tokenize(text, self.lowercase, self.ngram_range)

    def fit(self, texts: List[str]) -> "TfidfVectorizer":
        n = len(texts)
        self._n_docs = n
        df: Counter = Counter()
        for text in texts:
            for tok in set(self._tokens(text)):
                df[tok] += 1

        vocab = [t for t, c in df.items() if c >= self.min_df]
        vocab.sort(key=lambda t: -df[t])
        if self.max_features is not None:
            vocab = vocab[: self.max_features]

        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}
        self._idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in vocab}
        return self

    def transform(self, texts: List[str]) -> List[Dict[int, float]]:
        result: List[Dict[int, float]] = []
        for text in texts:
            tokens = self._tokens(text)
            if not tokens:
                result.append({})
                continue
            tf: Counter = Counter(tokens)
            vec: Dict[int, float] = {}
            norm = 0.0
            for tok, cnt in tf.items():
                if tok in self.vocabulary_:
                    idx = self.vocabulary_[tok]
                    val = (cnt / len(tokens)) * self._idf[tok]
                    vec[idx] = val
                    norm += val * val
            if norm > 0:
                sq = math.sqrt(norm)
                vec = {k: v / sq for k, v in vec.items()}
            result.append(vec)
        return result

    def fit_transform(self, texts: List[str]) -> List[Dict[int, float]]:
        self.fit(texts)
        return self.transform(texts)


# ---------------------------------------------------------------------------
# Logistic regression (batch gradient descent, binary)
# ---------------------------------------------------------------------------

class LogisticRegression:
    def __init__(
        self,
        max_iter: int = 200,
        C: float = 1.0,
        class_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight or {}
        self.classes_: List[str] = []
        self._weights: List[float] = []
        self._bias: float = 0.0
        self._n_features: int = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-500.0, min(500.0, x))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def fit(self, X: List[Dict[int, float]], y: List[str]) -> "LogisticRegression":
        self.classes_ = sorted(set(y))
        pos_class = self.classes_[1]

        n = len(X)
        n_features = max((max(x.keys(), default=0) for x in X if x), default=0) + 1
        self._n_features = n_features
        weights = [0.0] * n_features
        bias = 0.0

        binary_y = [1.0 if yi == pos_class else 0.0 for yi in y]
        sample_w = [self.class_weight.get(yi, 1.0) for yi in y]
        reg = 1.0 / (self.C * n)
        lr = 0.5
        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            # Forward pass
            preds: List[float] = []
            for xi in X:
                z = bias
                for idx, val in xi.items():
                    if idx < n_features:
                        z += weights[idx] * val
                preds.append(self._sigmoid(z))

            # Loss (cross-entropy)
            loss = sum(
                -y_i * math.log(max(p, 1e-15)) - (1 - y_i) * math.log(max(1 - p, 1e-15))
                for p, y_i in zip(preds, binary_y)
            ) / n

            # Gradients
            errs = [(p - yi) * sw for p, yi, sw in zip(preds, binary_y, sample_w)]
            bias -= lr * sum(errs) / n

            grad = [reg * w for w in weights]
            for i, xi in enumerate(X):
                scale = errs[i] / n
                for idx, val in xi.items():
                    if idx < n_features:
                        grad[idx] += scale * val
            for j in range(n_features):
                weights[j] -= lr * grad[j]

            if abs(prev_loss - loss) < 1e-7:
                break
            prev_loss = loss
            if (iteration + 1) % 100 == 0:
                lr *= 0.7

        self._weights = weights
        self._bias = bias
        return self

    def predict_proba(self, X: List[Dict[int, float]]) -> List[List[float]]:
        result: List[List[float]] = []
        for xi in X:
            z = self._bias
            for idx, val in xi.items():
                if idx < self._n_features:
                    z += self._weights[idx] * val
            p = self._sigmoid(z)
            result.append([1.0 - p, p])
        return result

    def predict(self, X: List[Dict[int, float]]) -> List[str]:
        return [
            self.classes_[1] if row[1] >= 0.5 else self.classes_[0]
            for row in self.predict_proba(X)
        ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self.steps = steps
        self._vec: TfidfVectorizer = steps[0][1]
        self._clf: LogisticRegression = steps[1][1]
        self._token_scaler = TokenScaler()

    @property
    def classes_(self) -> List[str]:
        return self._clf.classes_

    def _features(self, rows: List[Dict[int, float]], tokens: Optional[List[Any]]):
        # getattr keeps models pickled before the token feature was added working:
        # without a scaler they classify on text features only, as before.
        scaler = getattr(self, "_token_scaler", None)
        if scaler is None:
            return rows
        return _augment_sparse_rows(rows, tokens, scaler, len(self._vec.vocabulary_))

    def fit(self, texts: List[str], labels: List[str], tokens: Optional[List[Any]] = None) -> "Pipeline":
        rows = self._vec.fit_transform(texts)
        self._token_scaler.fit(_as_tokens(tokens, len(rows)))
        self._clf.fit(self._features(rows, tokens), labels)
        return self

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        return self._clf.predict(self._features(self._vec.transform(texts), tokens))

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        return self._clf.predict_proba(self._features(self._vec.transform(texts), tokens))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def train_test_split(
    *arrays: List,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[List] = None,
) -> List:
    rng = random.Random(random_state)
    n = len(arrays[0])

    if stratify is not None:
        class_idx: Dict[Any, List[int]] = {}
        for i, label in enumerate(stratify):
            class_idx.setdefault(label, []).append(i)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for indices in class_idx.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            n_test = max(1, round(len(shuffled) * test_size))
            test_idx.extend(shuffled[:n_test])
            train_idx.extend(shuffled[n_test:])
    else:
        indices = list(range(n))
        rng.shuffle(indices)
        n_test = max(1, round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    result: List = []
    for arr in arrays:
        result.append([arr[i] for i in train_idx])
        result.append([arr[i] for i in test_idx])
    return result


def accuracy_score(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


# ---------------------------------------------------------------------------
# Serialisation (pure pickle — no compiled deps)
# ---------------------------------------------------------------------------

def dump(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# LightGBM pipelines (lazy-import lightgbm/numpy so pure-Python path still works)
# ---------------------------------------------------------------------------

def _sparse_to_numpy(X: List[Dict[int, float]], n_features: int) -> Any:
    import numpy as np
    arr = np.zeros((len(X), n_features), dtype=np.float32)
    for i, xi in enumerate(X):
        for idx, val in xi.items():
            if idx < n_features:
                arr[i, idx] = val
    return arr


def _lgbm_predict(clf: Any, X: Any) -> Any:
    """Predict while suppressing sklearn feature-name mismatch warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        return clf.predict(X)


def _lgbm_predict_proba(clf: Any, X: Any) -> Any:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        return clf.predict_proba(X)


class LightGBMTfidfPipeline:
    """TF-IDF sparse features → LightGBM classifier."""

    def __init__(
        self,
        vectorizer: "TfidfVectorizer",
        class_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        self._vec = vectorizer
        self._class_weight = class_weight or {}
        self._clf: Any = None
        self.classes_: List[str] = []
        self._token_scaler = TokenScaler()

    def _matrix(self, rows: List[Dict[int, float]], tokens: Optional[List[Any]]):
        scaler = getattr(self, "_token_scaler", None)
        n_features = len(self._vec.vocabulary_)
        if scaler is None:
            return _sparse_to_numpy(rows, n_features)
        rows = _augment_sparse_rows(rows, tokens, scaler, n_features)
        return _sparse_to_numpy(rows, n_features + 1)

    def fit(self, texts: List[str], labels: List[str], tokens: Optional[List[Any]] = None) -> "LightGBMTfidfPipeline":
        import lightgbm as lgb
        rows = self._vec.fit_transform(texts)
        self._token_scaler.fit(_as_tokens(tokens, len(rows)))
        X = self._matrix(rows, tokens)
        self._clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight=self._class_weight or None,
            verbose=-1,
        )
        self._clf.fit(X, labels)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        return list(_lgbm_predict(self._clf, self._matrix(self._vec.transform(texts), tokens)))

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        return _lgbm_predict_proba(self._clf, self._matrix(self._vec.transform(texts), tokens)).tolist()


class LightGBMEmbeddingPipeline:
    """Embedding dense features → LightGBM classifier."""

    def __init__(
        self,
        vectorizer: "EmbeddingVectorizer",
        class_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        self._vec = vectorizer
        self._class_weight = class_weight or {}
        self._clf: Any = None
        self.classes_: List[str] = []
        self._token_scaler = TokenScaler()

    def _matrix(self, rows: List[List[float]], tokens: Optional[List[Any]]):
        import numpy as np
        scaler = getattr(self, "_token_scaler", None)
        if scaler is not None:
            rows = _augment_dense_rows(rows, tokens, scaler)
        return np.array(rows, dtype=np.float32)

    def fit(self, texts: List[str], labels: List[str], tokens: Optional[List[Any]] = None) -> "LightGBMEmbeddingPipeline":
        import lightgbm as lgb
        rows = self._vec.fit_transform(texts)
        self._token_scaler.fit(_as_tokens(tokens, len(rows)))
        X = self._matrix(rows, tokens)
        self._clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight=self._class_weight or None,
            verbose=-1,
        )
        self._clf.fit(X, labels)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        return list(_lgbm_predict(self._clf, self._matrix(self._vec.transform(texts), tokens)))

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        return _lgbm_predict_proba(self._clf, self._matrix(self._vec.transform(texts), tokens)).tolist()

    def prepare_for_inference(self) -> None:
        self._vec.prepare_for_inference()

    def warm_inference(self, prompt: str) -> bool:
        return self._vec.warm_inference(
            prompt,
            lambda: (
                self.predict([prompt], [None]),
                self.predict_proba([prompt], [None]),
            ),
        )


# ---------------------------------------------------------------------------
# Embedding vectoriser (calls OpenAI-compatible API, caches to disk)
# ---------------------------------------------------------------------------

class EmbeddingVectorizer:
    """Produces remote OpenAI or local SentenceTransformer embeddings."""

    # 8192 token limit; ~4 chars/token → 20 000 chars is safely under it
    MAX_CHARS = 20_000

    # Strict per-request timeout (seconds) for the embeddings API call. Defined
    # as a class attribute so models pickled before this field was added still
    # inherit it after unpickling.
    REQUEST_TIMEOUT = 30.0

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        cache_path: Optional["str | Path"] = None,
        base_url: str = "https://api.openai.com/v1",
        provider: str = "openai",
        task_prefix: str = "",
        local_device: str = "cpu",
        local_batch_size: int = 16,
        model_revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.cache_path = Path(cache_path) if cache_path else None
        self.base_url = base_url.rstrip("/")
        self.provider = provider.strip().lower()
        if self.provider not in {"openai", "local"}:
            raise ValueError(
                f"Unsupported embedding provider {provider!r}; expected 'openai' or 'local'."
            )
        self.task_prefix = task_prefix
        self.local_device = local_device
        self.local_batch_size = max(1, int(local_batch_size))
        self.model_revision = model_revision
        self.trust_remote_code = bool(trust_remote_code)
        self.persist_cache = True
        self.embedding_dim_: int = 0
        self._cache: Dict[str, List[float]] = {}
        self._local_encoder = None
        self._local_encoder_lock = threading.Lock()
        self._cache_lock = threading.RLock()

        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # The model can be hundreds of MB and is already available from the
        # Hugging Face cache. Persist only its identity, never the loaded object.
        state["_local_encoder"] = None
        state.pop("_local_encoder_lock", None)
        state.pop("_cache_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        provider = state.setdefault("provider", "openai")
        state.setdefault("task_prefix", "")
        state.setdefault("local_device", "cpu")
        state.setdefault("local_batch_size", 16)
        state.setdefault("model_revision", None)
        # Local artifacts produced before this setting existed used remote code.
        state.setdefault("trust_remote_code", provider == "local")
        state.setdefault("persist_cache", True)
        self.__dict__.update(state)
        self._local_encoder = None
        self._local_encoder_lock = threading.Lock()
        self._cache_lock = threading.RLock()

    def _trunc(self, text: str) -> str:
        return text[: self.MAX_CHARS]

    def _cache_key(self, text: str) -> str:
        truncated = self._trunc(text)
        if self.provider == "openai":
            # Preserve compatibility with the existing on-disk OpenAI cache.
            return truncated
        revision = self.model_revision or "main"
        return f"local:{self.model}:{revision}:{self.task_prefix}:{truncated}"

    def _save_cache(self) -> None:
        if self.persist_cache and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def prepare_for_inference(self) -> None:
        """Keep repeat-request caching in memory without persisting user prompts."""
        self.persist_cache = False

    def _missing_texts(self, texts: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for text in texts:
            key = self._cache_key(text)
            if key not in self._cache and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    def _fetch_openai(self, to_fetch: List[str]) -> None:
        if not to_fetch:
            return

        batch_size = 100
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i : i + batch_size]
            truncated = [self._trunc(t) for t in batch]
            payload = _json.dumps({"input": truncated, "model": self.model}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
                    data = _json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()
                logger.error("Embeddings API error %s: %s", exc.code, detail)
                raise RuntimeError(
                    f"Embeddings API error {exc.code}: {detail}"
                ) from exc

            for item in data["data"]:
                self._cache[self._cache_key(batch[item["index"]])] = item["embedding"]

            logger.info("Fetched embeddings %d/%d", i + len(batch), len(to_fetch))

    def _load_local_encoder(self) -> Any:
        encoder = self._local_encoder
        if encoder is not None:
            return encoder
        with self._local_encoder_lock:
            if self._local_encoder is None:
                if self.trust_remote_code and not self.model_revision:
                    raise RuntimeError(
                        "Local embedding models using remote code require a pinned "
                        "embedding_model_revision."
                    )
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "Local embeddings require sentence-transformers. "
                        "Install the project requirements."
                    ) from exc
                logger.info(
                    "Loading local embedding model '%s' on %s",
                    self.model,
                    self.local_device,
                )
                self._local_encoder = SentenceTransformer(
                    self.model,
                    device=self.local_device,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.model_revision,
                )
        return self._local_encoder

    def warmup(self) -> None:
        """Load the local model before the service begins accepting requests."""
        if self.provider == "local":
            self._load_local_encoder()

    def warm_inference(self, text: str, infer: Callable[[], Any]) -> bool:
        """Warm a local inference path without retaining the synthetic prompt."""
        if self.provider != "local":
            return False

        self.warmup()
        key = self._cache_key(text)
        with self._cache_lock:
            had_cached_value = key in self._cache
            cached_value = self._cache.pop(key, None)
            try:
                infer()
            finally:
                if had_cached_value:
                    self._cache[key] = cached_value
                else:
                    self._cache.pop(key, None)
        return True

    def _fetch_local(self, to_fetch: List[str]) -> None:
        if not to_fetch:
            return
        encoder = self._load_local_encoder()
        inputs = [f"{self.task_prefix}{self._trunc(text)}" for text in to_fetch]
        embeddings = encoder.encode(
            inputs,
            batch_size=self.local_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, embedding in zip(to_fetch, embeddings):
            self._cache[self._cache_key(text)] = embedding.tolist()
        logger.info("Generated %d local embeddings", len(to_fetch))

    def _fetch(self, texts: List[str]) -> None:
        """Generate and cache any texts not already cached."""
        with self._cache_lock:
            to_fetch = self._missing_texts(texts)
            if not to_fetch:
                return
            if self.provider == "local":
                self._fetch_local(to_fetch)
            else:
                self._fetch_openai(to_fetch)
            self._save_cache()

    def fit(self, texts: List[str]) -> "EmbeddingVectorizer":
        self._fetch(texts)
        if texts:
            self.embedding_dim_ = len(self._cache[self._cache_key(texts[0])])
        return self

    def transform(self, texts: List[str]) -> List[List[float]]:
        self._fetch(texts)
        return [self._cache[self._cache_key(t)] for t in texts]

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        self.fit(texts)
        return self.transform(texts)


# ---------------------------------------------------------------------------
# Dense logistic regression (batch gradient descent, for embedding vectors)
# ---------------------------------------------------------------------------

class DenseLogisticRegression:
    """Logistic regression for dense float vectors (e.g. embeddings)."""

    def __init__(
        self,
        max_iter: int = 300,
        C: float = 1.0,
        class_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight or {}
        self.classes_: List[str] = []
        self._weights: List[float] = []
        self._bias: float = 0.0
        self._n_features: int = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-500.0, min(500.0, x))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def fit(self, X: List[List[float]], y: List[str]) -> "DenseLogisticRegression":
        self.classes_ = sorted(set(y))
        pos_class = self.classes_[1]

        n = len(X)
        d = len(X[0]) if X else 0
        self._n_features = d
        weights = [0.0] * d
        bias = 0.0

        binary_y = [1.0 if yi == pos_class else 0.0 for yi in y]
        sample_w = [self.class_weight.get(yi, 1.0) for yi in y]
        reg = 1.0 / (self.C * n)
        lr = 0.1
        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            # Forward pass
            preds: List[float] = [
                self._sigmoid(bias + sum(w * x for w, x in zip(weights, xi)))
                for xi in X
            ]

            # Cross-entropy loss
            loss = sum(
                -yi * math.log(max(p, 1e-15)) - (1 - yi) * math.log(max(1 - p, 1e-15))
                for p, yi in zip(preds, binary_y)
            ) / n

            # Gradients (iterate over samples to build column-wise gradient)
            errs = [(p - yi) * sw for p, yi, sw in zip(preds, binary_y, sample_w)]
            bias -= lr * sum(errs) / n

            grad = [reg * w for w in weights]
            for i, xi in enumerate(X):
                scale = errs[i] / n
                for j in range(d):
                    grad[j] += scale * xi[j]
            for j in range(d):
                weights[j] -= lr * grad[j]

            if abs(prev_loss - loss) < 1e-7:
                break
            prev_loss = loss
            if (iteration + 1) % 100 == 0:
                lr *= 0.7

        self._weights = weights
        self._bias = bias
        return self

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        result: List[List[float]] = []
        for xi in X:
            z = self._bias + sum(w * x for w, x in zip(self._weights, xi))
            p = self._sigmoid(z)
            result.append([1.0 - p, p])
        return result

    def predict(self, X: List[List[float]]) -> List[str]:
        return [
            self.classes_[1] if row[1] >= 0.5 else self.classes_[0]
            for row in self.predict_proba(X)
        ]


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    def __init__(
        self,
        vectorizer: EmbeddingVectorizer,
        classifier: DenseLogisticRegression,
    ) -> None:
        self._vec = vectorizer
        self._clf = classifier
        self._token_scaler = TokenScaler()

    @property
    def classes_(self) -> List[str]:
        return self._clf.classes_

    def _features(self, rows: List[List[float]], tokens: Optional[List[Any]]):
        scaler = getattr(self, "_token_scaler", None)
        if scaler is None:
            return rows
        return _augment_dense_rows(rows, tokens, scaler)

    def fit(self, texts: List[str], labels: List[str], tokens: Optional[List[Any]] = None) -> "EmbeddingPipeline":
        rows = self._vec.fit_transform(texts)
        self._token_scaler.fit(_as_tokens(tokens, len(rows)))
        self._clf.fit(self._features(rows, tokens), labels)
        return self

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        return self._clf.predict(self._features(self._vec.transform(texts), tokens))

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        return self._clf.predict_proba(self._features(self._vec.transform(texts), tokens))

    def prepare_for_inference(self) -> None:
        self._vec.prepare_for_inference()

    def warm_inference(self, prompt: str) -> bool:
        return self._vec.warm_inference(
            prompt,
            lambda: (
                self.predict([prompt], [None]),
                self.predict_proba([prompt], [None]),
            ),
        )


class TunedEmbeddingPipeline:
    """EmbeddingVectorizer + any sklearn-compatible classifier; picklable."""

    def __init__(
        self,
        vectorizer: "EmbeddingVectorizer",
        classifier: Any,
        token_scaler: Optional["TokenScaler"] = None,
    ) -> None:
        self._vec = vectorizer
        self._clf = classifier
        self._token_scaler = token_scaler
        self.classes_: List[str] = list(classifier.classes_)

    def _matrix(self, texts: List[str], tokens: Optional[List[Any]]):
        import numpy as np
        rows = self._vec.transform(texts)
        scaler = getattr(self, "_token_scaler", None)
        if scaler is not None:
            rows = _augment_dense_rows(rows, tokens, scaler)
        return np.array(rows, dtype=np.float32)

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        return list(_lgbm_predict(self._clf, self._matrix(texts, tokens)))

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        return _lgbm_predict_proba(self._clf, self._matrix(texts, tokens)).tolist()

    def prepare_for_inference(self) -> None:
        self._vec.prepare_for_inference()

    def warm_inference(self, prompt: str) -> bool:
        return self._vec.warm_inference(
            prompt,
            lambda: (
                self.predict([prompt], [None]),
                self.predict_proba([prompt], [None]),
            ),
        )


class EnsembleEmbeddingPipeline:
    """EmbeddingVectorizer + soft-voting ensemble of sklearn-compatible classifiers."""

    def __init__(
        self,
        vectorizer: "EmbeddingVectorizer",
        classifiers: List[Tuple[str, Any]],
        token_scaler: Optional["TokenScaler"] = None,
    ) -> None:
        self._vec = vectorizer
        self._clfs = classifiers  # [(name, fitted_clf), ...]
        self._token_scaler = token_scaler
        self.classes_: List[str] = list(classifiers[0][1].classes_)

    def predict_proba(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[List[float]]:
        import numpy as np
        rows = self._vec.transform(texts)
        scaler = getattr(self, "_token_scaler", None)
        if scaler is not None:
            rows = _augment_dense_rows(rows, tokens, scaler)
        X = np.array(rows, dtype=np.float32)
        all_probas = [_lgbm_predict_proba(clf, X) for _, clf in self._clfs]
        avg = np.mean(all_probas, axis=0)
        return avg.tolist()

    def predict(self, texts: List[str], tokens: Optional[List[Any]] = None) -> List[str]:
        import numpy as np
        avg = np.array(self.predict_proba(texts, tokens))
        return [self.classes_[int(i)] for i in np.argmax(avg, axis=1)]

    def prepare_for_inference(self) -> None:
        self._vec.prepare_for_inference()

    def warm_inference(self, prompt: str) -> bool:
        return self._vec.warm_inference(
            prompt,
            lambda: (
                self.predict([prompt], [None]),
                self.predict_proba([prompt], [None]),
            ),
        )
