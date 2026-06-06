import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    DEFAULT_EMBEDDING_CACHE_PATH,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH,
    DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH,
    DEFAULT_LGBM_MODEL_PATH,
    DEFAULT_MODEL_PATH,
)
from app.modeling import (
    train_model,
    train_model_embeddings,
    train_model_ensemble_embeddings,
    train_model_lgbm,
    train_model_lgbm_embeddings,
    train_model_lgbm_embeddings_optuna,
    train_model_lgbm_embeddings_tuned,
)

ALL_VARIANTS = [
    ("TF-IDF + LogReg",                         "tfidf_logreg"),
    ("TF-IDF + LightGBM",                       "tfidf_lgbm"),
    ("Embeddings + LogReg",                     "emb_logreg"),
    ("Embeddings + LightGBM",                   "emb_lgbm"),
    ("Embeddings + LightGBM (tuned)",           "emb_lgbm_tuned"),
    ("Embeddings + LightGBM (Optuna)",          "emb_lgbm_optuna"),
    ("Embeddings + Ensemble (LGBM+XGB+CB)",     "emb_ensemble"),
]


def _print_result(label: str, result: dict) -> None:
    print(f"\n{'=' * 52}")
    print(f"  {label}")
    print(f"{'=' * 52}")
    print(f"  Model path:             {result['model_path']}")
    print(f"  Total examples:         {result['total_examples']}")
    print(f"  Label counts:           {result['label_counts']}")
    acc = result["validation_accuracy"]
    fcr = result["validation_false_cheap_rate"]
    print(f"  Validation accuracy:    {acc:.4f}" if acc is not None else "  Validation accuracy:    n/a")
    print(f"  Validation false-cheap: {fcr:.4f}" if fcr is not None else "  Validation false-cheap: n/a")
    if "best_params" in result:
        if "best_cv_scores" in result:
            for name, score in result["best_cv_scores"].items():
                print(f"  Best CV score ({name:<8}): {score:.4f}")
        else:
            print(f"  Best CV score:          {result['best_cv_score']:.4f}")
        print(f"  Best params:            {result['best_params']}")


def _run(variant: str, args: argparse.Namespace) -> dict:
    if variant == "tfidf_logreg":
        return train_model(args.input, args.model_output)
    if variant == "tfidf_lgbm":
        return train_model_lgbm(args.input, args.lgbm_output)
    if variant == "emb_logreg":
        return train_model_embeddings(args.input, args.embeddings_output, args.embeddings_cache)
    if variant == "emb_lgbm":
        return train_model_lgbm_embeddings(args.input, args.lgbm_embeddings_output, args.embeddings_cache)
    if variant == "emb_lgbm_tuned":
        return train_model_lgbm_embeddings_tuned(
            args.input, args.lgbm_embeddings_tuned_output, args.embeddings_cache,
            n_iter=args.tuning_iter, cv=args.tuning_cv,
        )
    if variant == "emb_lgbm_optuna":
        return train_model_lgbm_embeddings_optuna(
            args.input, args.lgbm_embeddings_optuna_output, args.embeddings_cache,
            n_trials=args.optuna_trials, cv=args.optuna_cv,
        )
    if variant == "emb_ensemble":
        return train_model_ensemble_embeddings(
            args.input, args.ensemble_embeddings_output, args.embeddings_cache,
            cv=args.ensemble_cv,
        )
    raise ValueError(variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to labeled JSON")
    parser.add_argument("--model-output",                default=str(DEFAULT_MODEL_PATH),                    help="TF-IDF + LogReg output path")
    parser.add_argument("--lgbm-output",                 default=str(DEFAULT_LGBM_MODEL_PATH),               help="TF-IDF + LightGBM output path")
    parser.add_argument("--embeddings-output",           default=str(DEFAULT_EMBEDDING_MODEL_PATH),          help="Embeddings + LogReg output path")
    parser.add_argument("--lgbm-embeddings-output",      default=str(DEFAULT_LGBM_EMBEDDING_MODEL_PATH),     help="Embeddings + LightGBM output path")
    parser.add_argument("--lgbm-embeddings-tuned-output", default=str(DEFAULT_LGBM_EMBEDDING_TUNED_MODEL_PATH),  help="Embeddings + LightGBM (tuned) output path")
    parser.add_argument("--lgbm-embeddings-optuna-output",default=str(DEFAULT_LGBM_EMBEDDING_OPTUNA_MODEL_PATH), help="Embeddings + LightGBM (Optuna) output path")
    parser.add_argument("--ensemble-embeddings-output",   default=str(DEFAULT_ENSEMBLE_EMBEDDING_MODEL_PATH),    help="Embeddings + Ensemble output path")
    parser.add_argument("--embeddings-cache",             default=str(DEFAULT_EMBEDDING_CACHE_PATH),             help="Shared embedding cache path")
    parser.add_argument("--tuning-iter",   type=int, default=25, help="Random search iterations (tuned variant)")
    parser.add_argument("--tuning-cv",     type=int, default=5,  help="CV folds (tuned variant)")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--optuna-cv",     type=int, default=5,  help="CV folds (Optuna variant)")
    parser.add_argument("--ensemble-cv",   type=int, default=5,  help="CV folds (ensemble grid search)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-embeddings",            action="store_true", help="Train Embeddings + LogReg")
    group.add_argument("--use-lgbm",                  action="store_true", help="Train TF-IDF + LightGBM")
    group.add_argument("--use-lgbm-embeddings",       action="store_true", help="Train Embeddings + LightGBM")
    group.add_argument("--use-lgbm-embeddings-tuned", action="store_true", help="Train Embeddings + LightGBM (RandomSearch tuning)")
    group.add_argument("--use-lgbm-embeddings-optuna",action="store_true", help="Train Embeddings + LightGBM (Optuna TPE tuning)")
    group.add_argument("--use-ensemble-embeddings",   action="store_true", help="Train Embeddings + Ensemble (LGBM+XGB+CatBoost, grid search)")
    group.add_argument("--compare",                   action="store_true", help="Train all 7 variants and compare")

    args = parser.parse_args()

    if args.compare:
        results = {}
        for label, key in ALL_VARIANTS:
            print(f"\nTraining: {label} ...")
            results[key] = _run(key, args)
            _print_result(label, results[key])

        print(f"\n{'=' * 52}")
        print("  COMPARISON (vs TF-IDF + LogReg baseline)")
        print(f"{'=' * 52}")
        baseline_acc = results["tfidf_logreg"]["validation_accuracy"] or 0.0
        baseline_fcr = results["tfidf_logreg"]["validation_false_cheap_rate"] or 0.0
        for label, key in ALL_VARIANTS[1:]:
            acc = results[key]["validation_accuracy"] or 0.0
            fcr = results[key]["validation_false_cheap_rate"] or 0.0
            da = acc - baseline_acc
            df = fcr - baseline_fcr
            print(f"  {label:<32}  acc {da:+.4f}   false-cheap {df:+.4f}")

    elif args.use_ensemble_embeddings:
        _print_result("Embeddings + Ensemble (LGBM+XGB+CB)", _run("emb_ensemble", args))
    elif args.use_lgbm_embeddings_optuna:
        _print_result("Embeddings + LightGBM (Optuna)", _run("emb_lgbm_optuna", args))
    elif args.use_lgbm_embeddings_tuned:
        _print_result("Embeddings + LightGBM (tuned)", _run("emb_lgbm_tuned", args))
    elif args.use_lgbm:
        _print_result("TF-IDF + LightGBM", _run("tfidf_lgbm", args))
    elif args.use_lgbm_embeddings:
        _print_result("Embeddings + LightGBM", _run("emb_lgbm", args))
    elif args.use_embeddings:
        _print_result("Embeddings + LogReg", _run("emb_logreg", args))
    else:
        _print_result("TF-IDF + LogReg", _run("tfidf_logreg", args))


if __name__ == "__main__":
    main()
