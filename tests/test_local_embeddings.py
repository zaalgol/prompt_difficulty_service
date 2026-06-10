import pickle
import threading
import time

import numpy as np
import pytest

from app import ml
from app.ml import DenseLogisticRegression, EmbeddingPipeline, EmbeddingVectorizer
from app.modeling import load_model


class FakeLocalEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array(
            [[float(index), float(len(text))] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def test_local_embeddings_do_not_call_openai(monkeypatch):
    vectorizer = EmbeddingVectorizer(
        model="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        task_prefix="classification: ",
    )
    encoder = FakeLocalEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("local embeddings called OpenAI"),
    )

    rows = vectorizer.transform(["Route this prompt"])

    assert rows == [[0.0, 33.0]]
    texts, kwargs = encoder.calls[0]
    assert texts == ["classification: Route this prompt"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


def test_local_cache_is_namespaced_from_legacy_openai_cache(monkeypatch):
    text = "same prompt"
    vectorizer = EmbeddingVectorizer(
        model="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        task_prefix="classification: ",
    )
    vectorizer._cache[text] = [99.0]
    encoder = FakeLocalEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)

    rows = vectorizer.transform([text])

    assert rows == [[0.0, 27.0]]
    assert vectorizer._cache[text] == [99.0]
    assert vectorizer._cache_key(text) in vectorizer._cache


def test_local_cache_separates_models_and_task_prefixes():
    text = "same prompt"
    classification = EmbeddingVectorizer(
        model="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        task_prefix="classification: ",
    )
    search = EmbeddingVectorizer(
        model="nomic-ai/nomic-embed-text-v1.5",
        provider="local",
        task_prefix="search_query: ",
    )
    other_model = EmbeddingVectorizer(
        model="another/model",
        provider="local",
        task_prefix="classification: ",
    )

    assert len(
        {
            classification._cache_key(text),
            search._cache_key(text),
            other_model._cache_key(text),
        }
    ) == 3


def test_local_cache_separates_model_revisions():
    text = "same prompt"
    first = EmbeddingVectorizer(
        model="local/model",
        model_revision="revision-a",
        provider="local",
    )
    second = EmbeddingVectorizer(
        model="local/model",
        model_revision="revision-b",
        provider="local",
    )

    assert first._cache_key(text) != second._cache_key(text)


def test_duplicate_text_is_encoded_only_once(monkeypatch):
    vectorizer = EmbeddingVectorizer(model="local/model", provider="local")
    encoder = FakeLocalEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)

    rows = vectorizer.transform(["repeat", "repeat"])

    assert rows == [[0.0, 6.0], [0.0, 6.0]]
    assert encoder.calls[0][0] == ["repeat"]


def test_cached_local_embedding_skips_encoder(monkeypatch):
    vectorizer = EmbeddingVectorizer(model="local/model", provider="local")
    vectorizer._cache[vectorizer._cache_key("cached")] = [1.0, 2.0]
    monkeypatch.setattr(
        vectorizer,
        "_load_local_encoder",
        lambda: pytest.fail("cached embedding loaded local model"),
    )

    assert vectorizer.transform(["cached"]) == [[1.0, 2.0]]


def test_local_encoder_is_not_serialized(monkeypatch):
    vectorizer = EmbeddingVectorizer(model="local/model", provider="local")
    encoder = FakeLocalEncoder()
    vectorizer._local_encoder = encoder

    restored = pickle.loads(pickle.dumps(vectorizer))

    assert restored._local_encoder is None
    assert restored.provider == "local"
    assert restored.model == "local/model"
    assert restored._cache_lock is not None


def test_legacy_vectorizer_state_defaults_to_openai():
    vectorizer = EmbeddingVectorizer.__new__(EmbeddingVectorizer)
    vectorizer.__setstate__(
        {
            "model": "text-embedding-3-small",
            "api_key": "",
            "cache_path": None,
            "base_url": "https://api.openai.com/v1",
            "embedding_dim_": 0,
            "_cache": {"legacy": [1.0]},
        }
    )

    assert vectorizer.provider == "openai"
    assert vectorizer._cache_key("legacy") == "legacy"
    assert vectorizer.transform(["legacy"]) == [[1.0]]
    assert vectorizer.task_prefix == ""
    assert vectorizer.local_device == "cpu"
    assert vectorizer.local_batch_size == 16
    assert vectorizer.persist_cache is True


def test_invalid_embedding_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingVectorizer(provider="unknown")


def test_inference_mode_does_not_persist_new_prompt(tmp_path, monkeypatch):
    cache_path = tmp_path / "embeddings.pkl"
    vectorizer = EmbeddingVectorizer(
        model="local/model",
        provider="local",
        cache_path=cache_path,
    )
    encoder = FakeLocalEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)

    vectorizer.prepare_for_inference()
    assert vectorizer.transform(["private user prompt"])

    assert vectorizer.persist_cache is False
    assert not cache_path.exists()


def test_loading_artifact_automatically_disables_cache_persistence(tmp_path):
    vectorizer = EmbeddingVectorizer(
        model="local/model",
        provider="local",
        cache_path=tmp_path / "embeddings.pkl",
    )
    artifact_path = tmp_path / "model.joblib"
    ml.dump(
        {
            "pipeline": EmbeddingPipeline(
                vectorizer,
                DenseLogisticRegression(),
            )
        },
        artifact_path,
    )

    loaded = load_model(artifact_path)

    assert loaded["pipeline"]._vec.persist_cache is False


def test_warm_inference_removes_dummy_but_restores_existing_value(monkeypatch):
    vectorizer = EmbeddingVectorizer(model="local/model", provider="local")
    encoder = FakeLocalEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)
    key = vectorizer._cache_key("warmup")
    original = [9.0, 8.0]
    vectorizer._cache[key] = original

    warmed = vectorizer.warm_inference(
        "warmup", lambda: vectorizer.transform(["warmup"])
    )

    assert warmed is True
    assert vectorizer._cache[key] is original


def test_openai_warm_inference_never_calls_callback():
    vectorizer = EmbeddingVectorizer(provider="openai")

    assert vectorizer.warm_inference(
        "warmup", lambda: pytest.fail("OpenAI warmup made a synthetic request")
    ) is False


def test_remote_code_requires_pinned_revision():
    vectorizer = EmbeddingVectorizer(
        model="remote/model",
        provider="local",
        trust_remote_code=True,
    )

    with pytest.raises(RuntimeError, match="pinned embedding_model_revision"):
        vectorizer._load_local_encoder()


def test_concurrent_cache_misses_generate_embedding_once(monkeypatch):
    vectorizer = EmbeddingVectorizer(model="local/model", provider="local")

    class SlowEncoder(FakeLocalEncoder):
        def encode(self, texts, **kwargs):
            time.sleep(0.05)
            return super().encode(texts, **kwargs)

    encoder = SlowEncoder()
    monkeypatch.setattr(vectorizer, "_load_local_encoder", lambda: encoder)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(vectorizer.transform(["same prompt"]))
        )
        for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(encoder.calls) == 1
    assert results == [[[0.0, 11.0]]] * 4
