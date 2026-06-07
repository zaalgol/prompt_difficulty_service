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
uvicorn app.main:app --reload --port 8081
```

## Run tests

```powershell
pytest
```

## Health check

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/health"
```

## Classify one prompt via API

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8081/classify" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Refactor the authentication flow and explain the security tradeoffs"}'
```

Optionally include `total_input_tokens` (the prompt's input-token count). Models
trained with it use it as an auxiliary feature; omit it when unknown (it maps to
a neutral value) and models trained before the feature ignore it.

## Classify one prompt via CLI (no server needed)

```powershell
python scripts/classify_prompt.py --prompt "design a scalable auth system" --total-input-tokens 45000
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

## Anonymize a prompt via API

`/anonymize` detects PII and replaces it with realistic, consistent fakes before a
prompt is sent to an LLM (Presidio + Faker; see `app/presidio_service/`). Requires a
spaCy model — pinned in `requirements.txt`, so `pip install` covers it.

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8081/anonymize" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"I am John Smith, born 1960-04-12. When do I retire at 67?","session_id":"s1"}'
```

- `session_id` (optional): same value -> same fake across prompts in that session.
- `preserve_entity_types` (optional): force entity types to be kept unchanged.
- `auto_preserve` (default true): also auto-keep values the answer depends on
  (e.g. DATE_TIME for retirement/age questions). The example above keeps
  `1960-04-12` while anonymizing the name.

## Logging

Logs go to the terminal and `logs/service.log` (rotating). Default level: `INFO`.

```powershell
$env:LOG_LEVEL = "DEBUG"
```
