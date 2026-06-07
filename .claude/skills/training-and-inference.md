# Skill: Training and Inference

Use this when training or running the FastAPI service.

## Train model

Default (TF-IDF + LogReg baseline):

```powershell
python scripts/train_model.py --input data/report_labeled_binary.json
```

All variants (pass one flag):

| Flag | Model | Output path |
|------|-------|-------------|
| *(none)* | TF-IDF + LogReg | `models/prompt_classifier.joblib` |
| `--use-lgbm` | TF-IDF + LightGBM | `models/prompt_classifier_lgbm.joblib` |
| `--use-embeddings` | Embeddings + LogReg | `models/prompt_classifier_embeddings.joblib` |
| `--use-lgbm-embeddings` | Embeddings + LightGBM | `models/prompt_classifier_lgbm_embeddings.joblib` |
| `--use-lgbm-embeddings-tuned` | Embeddings + LightGBM (RandomSearch) | `models/prompt_classifier_lgbm_embeddings_tuned.joblib` |
| `--use-lgbm-embeddings-optuna` | Embeddings + LightGBM (Optuna TPE) | `models/prompt_classifier_lgbm_embeddings_optuna.joblib` |
| `--use-ensemble-embeddings` | Embeddings + LGBM + XGBoost + CatBoost | `models/prompt_classifier_ensemble_embeddings.joblib` |
| `--compare` | Train all 7 and print comparison table | all of the above |

Embedding variants require `OPENAI_API_KEY` to be set.

## Main metric

Prioritize low false cheap rate:

```text
hard/escalate prompt incorrectly predicted as cheap_ok
```

This is more important than general accuracy.

## Active model at runtime

`service_config.json` sets the model loaded at startup:

```json
{ "model_path": "models/prompt_classifier_ensemble_embeddings.joblib" }
```

Change this file to switch models without code changes.

## Run API

```powershell
uvicorn app.main:app --reload --port 8080
```

## Run tests

```powershell
pytest
```

## Health check

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/health"
```

## Classify one prompt via API

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8080/classify" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Refactor the authentication flow and explain the security tradeoffs"}'
```

## Classify one prompt via CLI (no server needed)

```powershell
python scripts/classify_prompt.py --prompt "design a scalable auth system"
```

## Expected response fields

```json
{
  "label": "escalate",
  "confidence": 0.91,
  "model_version": "mvp-v1",
  "method": "trained_model",
  "reason": "...",
  "features": {}
}
```

If no trained model exists, `/classify` uses rule-based fallback (`method: "rule_based_fallback"`).

## Logging

Logs go to the terminal and `logs/service.log` (rotating). Default level: `INFO`.

```powershell
$env:LOG_LEVEL = "DEBUG"
```
