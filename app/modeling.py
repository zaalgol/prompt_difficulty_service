import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app import ml
from app.ml import (
    DenseLogisticRegression,
    EmbeddingPipeline,
    EmbeddingVectorizer,
    EnsembleEmbeddingPipeline,
    LightGBMEmbeddingPipeline,
    LightGBMTfidfPipeline,
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    TunedEmbeddingPipeline,
    accuracy_score,
    train_test_split,
)
from app.config import (
    DEFAULT_EMBEDDING_CACHE_PATH,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH,
    DEFAULT_LGBM_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    EMBEDDING_MODEL,
    LABEL_CHEAP_OK,
    LABEL_ESCALATE,
    MIN_CHEAP_CONFIDENCE,
    MODEL_VERSION,
)
from app.labeling import rule_based_label
from app.logging_config import get_logger

logger = get_logger(__name__)


def _timestamped_path(path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return path.parent / f"{ts}__{path.name}"


def _load_labeled_examples(path: str | Path) -> Tuple[List[str], List[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Input JSON must contain a top-level 'prompts' array.")

    texts: List[str] = []
    labels: List[str] = []

    for item in prompts:
        prompt = item.get("prompt")
        label = item.get("difficulty_label")
        if isinstance(prompt, str) and label in {LABEL_CHEAP_OK, LABEL_ESCALATE}:
            texts.append(prompt)
            labels.append(label)

    if not texts:
        raise ValueError("No labeled examples found. Expected 'prompt' and 'difficulty_label' fields.")

    if len(set(labels)) < 2:
        raise ValueError("Need at least two label classes to train a classifier.")

    return texts, labels


def train_model(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    texts, labels = _load_labeled_examples(labeled_json_path)
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000, min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3})),
        ]
    )
    return _train_with_pipeline(pipeline, texts, labels, Counter(labels), MODEL_VERSION, model_output_path)


def train_model_embeddings(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
    cache_path: str | Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> Dict[str, Any]:
    texts, labels = _load_labeled_examples(labeled_json_path)
    logger.info("Fetching embeddings for %d examples (cached at %s)", len(texts), cache_path)
    pipeline = EmbeddingPipeline(
        EmbeddingVectorizer(model=EMBEDDING_MODEL, cache_path=cache_path),
        DenseLogisticRegression(max_iter=300, class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3}),
    )
    return _train_with_pipeline(pipeline, texts, labels, Counter(labels), f"{MODEL_VERSION}-embeddings", model_output_path)


def _train_with_pipeline(
    pipeline: Any,
    texts: List[str],
    labels: List[str],
    label_counts: Counter,
    model_version: str,
    model_output_path: str | Path,
) -> Dict[str, Any]:
    """Shared train/validate/save logic for any pipeline object."""
    logger.info(
        "Training '%s' on %d examples (counts=%s)",
        model_version, len(texts), dict(label_counts),
    )
    validation_accuracy: Optional[float] = None
    validation_false_cheap_rate: Optional[float] = None

    if len(texts) >= 50 and min(label_counts.values()) >= 5:
        x_train, x_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels,
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        validation_accuracy = float(accuracy_score(y_val, y_pred))

        false_cheap = sum(
            1 for a, p in zip(y_val, y_pred)
            if a == LABEL_ESCALATE and p == LABEL_CHEAP_OK
        )
        true_escalate = sum(1 for a in y_val if a == LABEL_ESCALATE)
        validation_false_cheap_rate = false_cheap / true_escalate if true_escalate else 0.0
        logger.info(
            "Validation: accuracy=%.4f false_cheap_rate=%.4f",
            validation_accuracy, validation_false_cheap_rate,
        )
    else:
        logger.warning(
            "Too few examples for a holdout split (%d examples); "
            "training on all data without validation",
            len(texts),
        )
        pipeline.fit(texts, labels)

    artifact = {
        "model_version": model_version,
        "pipeline": pipeline,
        "labels": [LABEL_CHEAP_OK, LABEL_ESCALATE],
        "training_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
    }
    output_path = _timestamped_path(Path(model_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml.dump(artifact, output_path)
    logger.info("Saved model '%s' to %s", model_version, output_path)

    return {
        "model_path": str(output_path),
        "total_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
    }


def train_model_lgbm(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_LGBM_MODEL_PATH,
) -> Dict[str, Any]:
    texts, labels = _load_labeled_examples(labeled_json_path)
    pipeline = LightGBMTfidfPipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000, min_df=2),
        class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
    )
    return _train_with_pipeline(
        pipeline, texts, labels, Counter(labels),
        f"{MODEL_VERSION}-lgbm", model_output_path,
    )


def train_model_lgbm_embeddings(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_LGBM_EMBEDDING_MODEL_PATH,
    cache_path: str | Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> Dict[str, Any]:
    texts, labels = _load_labeled_examples(labeled_json_path)
    logger.info("Fetching embeddings for %d examples (cached at %s)", len(texts), cache_path)
    pipeline = LightGBMEmbeddingPipeline(
        EmbeddingVectorizer(model=EMBEDDING_MODEL, cache_path=cache_path),
        class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
    )
    return _train_with_pipeline(
        pipeline, texts, labels, Counter(labels),
        f"{MODEL_VERSION}-lgbm-embeddings", model_output_path,
    )


def train_model_lgbm_embeddings_tuned(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH,
    cache_path: str | Path = DEFAULT_EMBEDDING_CACHE_PATH,
    n_iter: int = 25,
    cv: int = 5,
) -> Dict[str, Any]:
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    texts, labels = _load_labeled_examples(labeled_json_path)
    label_counts = Counter(labels)

    logger.info("Fetching embeddings for %d examples (cached at %s)", len(texts), cache_path)
    vectorizer = EmbeddingVectorizer(model=EMBEDDING_MODEL, cache_path=cache_path)
    X_all = np.array(vectorizer.fit_transform(texts), dtype=np.float32)
    y_all = np.array(labels)

    # Hold out 15% for unbiased final evaluation
    splits = train_test_split(
        list(range(len(texts))), labels,
        test_size=0.15, random_state=42, stratify=labels,
    )
    train_idx, val_idx = splits[0], splits[1]
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    param_dist = {
        "n_estimators":       [100, 200, 300, 500, 800],
        "learning_rate":      [0.01, 0.03, 0.05, 0.1, 0.15],
        "num_leaves":         [15, 31, 63, 127],
        "min_child_samples":  [5, 10, 20, 30],
        "subsample":          [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":          [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda":         [0, 0.01, 0.1, 0.5, 1.0],
    }

    base_clf = lgb.LGBMClassifier(
        class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
        verbose=-1,
    )

    logger.info("Searching %d hyperparameter combinations × %d-fold CV", n_iter, cv)
    search = RandomizedSearchCV(
        base_clf,
        param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=make_scorer(balanced_accuracy_score),
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        search.fit(X_train, y_train)

    logger.info("Best CV balanced-accuracy: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    pipeline = TunedEmbeddingPipeline(vectorizer, search.best_estimator_)

    y_pred = pipeline.predict([texts[i] for i in val_idx])
    validation_accuracy = float(sum(a == b for a, b in zip(y_val.tolist(), y_pred)) / len(y_pred))
    true_escalate = sum(1 for a in y_val if a == LABEL_ESCALATE)
    false_cheap = sum(1 for a, p in zip(y_val.tolist(), y_pred) if a == LABEL_ESCALATE and p == LABEL_CHEAP_OK)
    validation_false_cheap_rate = false_cheap / true_escalate if true_escalate else 0.0

    artifact = {
        "model_version": f"{MODEL_VERSION}-lgbm-embeddings-tuned",
        "pipeline": pipeline,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "labels": [LABEL_CHEAP_OK, LABEL_ESCALATE],
        "training_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
    }
    output_path = _timestamped_path(Path(model_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml.dump(artifact, output_path)

    return {
        "model_path": str(output_path),
        "total_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
    }


def train_model_lgbm_embeddings_optuna(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH,
    cache_path: str | Path = DEFAULT_EMBEDDING_CACHE_PATH,
    n_trials: int = 50,
    cv: int = 5,
) -> Dict[str, Any]:
    import warnings
    import lightgbm as lgb
    import numpy as np
    import optuna
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    texts, labels = _load_labeled_examples(labeled_json_path)
    label_counts = Counter(labels)

    logger.info("Fetching embeddings for %d examples (cached at %s)", len(texts), cache_path)
    vectorizer = EmbeddingVectorizer(model=EMBEDDING_MODEL, cache_path=cache_path)
    X_all = np.array(vectorizer.fit_transform(texts), dtype=np.float32)
    y_all = np.array(labels)

    splits = train_test_split(
        list(range(len(texts))), labels,
        test_size=0.15, random_state=42, stratify=labels,
    )
    train_idx, val_idx = splits[0], splits[1]
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def objective(trial: "optuna.Trial") -> float:  # noqa: D401
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        clf = lgb.LGBMClassifier(
            **params,
            class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
            verbose=-1,
        )
        scores = []
        for fold_train_idx, fold_val_idx in kf.split(X_train, y_train):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                clf.fit(X_train[fold_train_idx], y_train[fold_train_idx])
                y_pred = clf.predict(X_train[fold_val_idx])
            scores.append(balanced_accuracy_score(y_train[fold_val_idx], y_pred))
        return sum(scores) / len(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    logger.info("Running Optuna TPE search: %d trials × %d-fold CV", n_trials, cv)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best CV balanced-accuracy: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    best_clf = lgb.LGBMClassifier(
        **study.best_params,
        class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        best_clf.fit(X_train, y_train)

    pipeline = TunedEmbeddingPipeline(vectorizer, best_clf)

    y_pred = pipeline.predict([texts[i] for i in val_idx])
    validation_accuracy = float(sum(a == b for a, b in zip(y_val.tolist(), y_pred)) / len(y_pred))
    true_escalate = sum(1 for a in y_val if a == LABEL_ESCALATE)
    false_cheap = sum(1 for a, p in zip(y_val.tolist(), y_pred) if a == LABEL_ESCALATE and p == LABEL_CHEAP_OK)
    validation_false_cheap_rate = false_cheap / true_escalate if true_escalate else 0.0

    artifact = {
        "model_version": f"{MODEL_VERSION}-lgbm-embeddings-optuna",
        "pipeline": pipeline,
        "best_params": study.best_params,
        "best_cv_score": study.best_value,
        "labels": [LABEL_CHEAP_OK, LABEL_ESCALATE],
        "training_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
    }
    output_path = _timestamped_path(Path(model_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml.dump(artifact, output_path)

    return {
        "model_path": str(output_path),
        "total_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
        "best_params": study.best_params,
        "best_cv_score": study.best_value,
    }


def _grid_size(grid: Dict[str, list]) -> int:
    result = 1
    for v in grid.values():
        result *= len(v)
    return result


def train_model_ensemble_embeddings(
    labeled_json_path: str | Path,
    model_output_path: str | Path = DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH,
    cache_path: str | Path = DEFAULT_EMBEDDING_CACHE_PATH,
    cv: int = 5,
) -> Dict[str, Any]:
    import warnings
    import lightgbm as lgb
    import numpy as np
    import xgboost as xgb
    import catboost as cb
    from sklearn.metrics import balanced_accuracy_score, make_scorer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    texts, labels = _load_labeled_examples(labeled_json_path)
    label_counts = Counter(labels)

    logger.info("Fetching embeddings for %d examples (cached at %s)", len(texts), cache_path)
    vectorizer = EmbeddingVectorizer(model=EMBEDDING_MODEL, cache_path=cache_path)
    X_all = np.array(vectorizer.fit_transform(texts), dtype=np.float32)
    y_all = np.array(labels)

    splits = train_test_split(
        list(range(len(texts))), labels,
        test_size=0.15, random_state=42, stratify=labels,
    )
    train_idx, val_idx = splits[0], splits[1]
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scorer = make_scorer(balanced_accuracy_score)
    logger.info("Training ensemble (LightGBM + XGBoost + CatBoost) on %d examples", len(texts))

    # ── LightGBM ────────────────────────────────────────────────────────────
    lgbm_grid = {
        "n_estimators":      [200, 400],
        "learning_rate":     [0.05, 0.1],
        "num_leaves":        [15, 31],
        "min_child_samples": [5, 10],
    }
    logger.info("Grid-searching LightGBM (%d combos × %d folds)", _grid_size(lgbm_grid), cv)
    lgbm_search = GridSearchCV(
        lgb.LGBMClassifier(
            class_weight={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
            verbose=-1,
        ),
        lgbm_grid, cv=kf, scoring=scorer, n_jobs=1, verbose=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        lgbm_search.fit(X_train, y_train)
    logger.info("LGBM best CV balanced-acc: %.4f", lgbm_search.best_score_)
    logger.info("LGBM best params: %s", lgbm_search.best_params_)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    # XGBoost requires numeric labels; LabelEncoder sorts alphabetically so
    # cheap_ok→0, escalate→1, matching the column order of LGBM/CatBoost.
    from sklearn.preprocessing import LabelEncoder
    xgb_le = LabelEncoder().fit(y_train)
    y_train_xgb = xgb_le.transform(y_train)

    xgb_grid = {
        "n_estimators":  [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
        "subsample":     [0.8, 1.0],
    }
    logger.info("Grid-searching XGBoost (%d combos × %d folds)", _grid_size(xgb_grid), cv)
    xgb_search = GridSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=1.3,
            eval_metric="logloss",
            verbosity=0,
        ),
        xgb_grid, cv=kf, scoring=scorer, n_jobs=1, verbose=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        xgb_search.fit(X_train, y_train_xgb)
    logger.info("XGBoost best CV balanced-acc: %.4f", xgb_search.best_score_)
    logger.info("XGBoost best params: %s", xgb_search.best_params_)

    # ── CatBoost ─────────────────────────────────────────────────────────────
    # sklearn.clone() cannot clone CatBoost when class_weights is passed to the
    # constructor (CatBoost modifies the param internally). Search without it,
    # then refit the winner with class_weights applied.
    cat_grid = {
        "iterations":    [200, 400],
        "learning_rate": [0.05, 0.1],
        "depth":         [4, 6],
    }
    logger.info("Grid-searching CatBoost (%d combos × %d folds)", _grid_size(cat_grid), cv)
    cb_search = GridSearchCV(
        cb.CatBoostClassifier(verbose=0),
        cat_grid, cv=kf, scoring=scorer, n_jobs=1, verbose=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cb_search.fit(X_train, y_train)
    logger.info("CatBoost best CV balanced-acc: %.4f", cb_search.best_score_)
    logger.info("CatBoost best params: %s", cb_search.best_params_)

    # Refit best CatBoost params with class weighting on the full training set
    best_cb = cb.CatBoostClassifier(
        **cb_search.best_params_,
        class_weights={LABEL_CHEAP_OK: 1.0, LABEL_ESCALATE: 1.3},
        verbose=0,
    )
    best_cb.fit(X_train, y_train)

    # ── Soft-voting ensemble ─────────────────────────────────────────────────
    pipeline = EnsembleEmbeddingPipeline(
        vectorizer,
        [
            ("lgbm",     lgbm_search.best_estimator_),
            ("xgb",      xgb_search.best_estimator_),
            ("catboost", best_cb),
        ],
    )

    val_texts = [texts[i] for i in val_idx]
    y_pred = pipeline.predict(val_texts)
    validation_accuracy = float(sum(a == b for a, b in zip(y_val.tolist(), y_pred)) / len(y_pred))
    true_escalate = sum(1 for a in y_val if a == LABEL_ESCALATE)
    false_cheap = sum(1 for a, p in zip(y_val.tolist(), y_pred) if a == LABEL_ESCALATE and p == LABEL_CHEAP_OK)
    validation_false_cheap_rate = false_cheap / true_escalate if true_escalate else 0.0

    best_params = {
        "lgbm":     lgbm_search.best_params_,
        "xgb":      xgb_search.best_params_,
        "catboost": cb_search.best_params_,
    }
    best_cv_scores = {
        "lgbm":     lgbm_search.best_score_,
        "xgb":      xgb_search.best_score_,
        "catboost": cb_search.best_score_,
    }

    artifact = {
        "model_version": f"{MODEL_VERSION}-ensemble-embeddings",
        "pipeline": pipeline,
        "best_params": best_params,
        "best_cv_scores": best_cv_scores,
        "labels": [LABEL_CHEAP_OK, LABEL_ESCALATE],
        "training_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
    }
    output_path = _timestamped_path(Path(model_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml.dump(artifact, output_path)

    return {
        "model_path": str(output_path),
        "total_examples": len(texts),
        "label_counts": dict(label_counts),
        "validation_accuracy": validation_accuracy,
        "validation_false_cheap_rate": validation_false_cheap_rate,
        "best_params": best_params,
        "best_cv_scores": best_cv_scores,
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Optional[Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        return None
    return ml.load(path)


def classify_from_artifact(prompt: str, artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify a prompt using a pre-loaded model artifact (or rule-based fallback)."""
    if artifact is None:
        label, confidence, reason, features = rule_based_label(prompt)
        logger.info(
            "Classified prompt -> %s (confidence=%.4f, method=rule_based_fallback)",
            label, confidence,
        )
        return {
            "label": label,
            "confidence": confidence,
            "model_version": "rule-based-only",
            "method": "rule_based_fallback",
            "reason": reason,
            "features": features,
        }

    pipeline = artifact["pipeline"]
    predicted_label = pipeline.predict([prompt])[0]

    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        probabilities = pipeline.predict_proba([prompt])[0]
        class_to_probability = dict(zip(classes, probabilities))
        confidence = float(class_to_probability[predicted_label])
    else:
        confidence = 0.70

    final_label = predicted_label
    reason = "Classified by trained lightweight model."

    if predicted_label == LABEL_CHEAP_OK and confidence < MIN_CHEAP_CONFIDENCE:
        final_label = LABEL_ESCALATE
        reason = (
            "Model predicted cheap_ok, but confidence was below the conservative threshold; "
            "routing label changed to escalate."
        )
        logger.info(
            "Confidence %.4f below threshold %.2f; overriding cheap_ok -> escalate",
            confidence, MIN_CHEAP_CONFIDENCE,
        )

    _, _, rule_reason, features = rule_based_label(prompt)

    logger.info(
        "Classified prompt -> %s (confidence=%.4f, raw=%s, model=%s)",
        final_label, confidence, predicted_label, artifact.get("model_version", MODEL_VERSION),
    )

    return {
        "label": final_label,
        "confidence": round(confidence, 4),
        "model_version": artifact.get("model_version", MODEL_VERSION),
        "method": "trained_model",
        "reason": reason,
        "features": {
            **features,
            "raw_model_label": predicted_label,
            "rule_based_reason": rule_reason,
            "cheap_confidence_threshold": MIN_CHEAP_CONFIDENCE,
        },
    }


def classify_with_model(prompt: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    return classify_from_artifact(prompt, load_model(model_path))
