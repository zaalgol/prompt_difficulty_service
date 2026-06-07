# Prompt Difficulty Classification Service

MVP FastAPI service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

Because there is no human-labeled dataset yet, the first stage uses rule-based pseudo-labeling.
Then a lightweight classifier can be trained on the pseudo-labeled dataset.

## Install

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run API

```powershell
uvicorn app.main:app --reload --port 8080
```

## Run tests

```powershell
pytest
```

## Label a dataset from CLI

```powershell
python scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

## Train a model from CLI

Default (TF-IDF + LogReg baseline):

```powershell
python scripts/train_model.py --input data/report_labeled_binary.json
```

Other variants (pass one flag):

| Flag | Model |
|------|-------|
| *(none)* | TF-IDF + LogReg |
| `--use-lgbm` | TF-IDF + LightGBM |
| `--use-embeddings` | Embeddings + LogReg |
| `--use-lgbm-embeddings` | Embeddings + LightGBM |
| `--use-lgbm-embeddings-tuned` | Embeddings + LightGBM (RandomSearch) |
| `--use-lgbm-embeddings-optuna` | Embeddings + LightGBM (Optuna TPE) |
| `--use-ensemble-embeddings` | Embeddings + LGBM + XGBoost + CatBoost |
| `--compare` | Train all 7 variants and print a comparison table |

Embedding variants require `OPENAI_API_KEY` to be set.

## Classify a prompt

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8080/classify" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Refactor the authentication flow and explain the security tradeoffs"}'
```

curl:

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor this authentication flow and explain the tradeoffs"}'
```

## Classify from CLI (no server needed)

```powershell
python scripts/classify_prompt.py --prompt "design a scalable auth system"
```

## Main API endpoints

- `GET /health`
- `POST /classify`
- `POST /label-dataset`
- `POST /train`

## Logging

Logs are written to both the terminal and `logs/service.log` (rotating, 5 MB × 5 files).
Default level is `INFO`. Override with the `LOG_LEVEL` environment variable:

```powershell
$env:LOG_LEVEL = "DEBUG"
uvicorn app.main:app --reload --port 8080
```

## Active model

The model loaded at startup is set in `service_config.json`:

```json
{ "model_path": "models/prompt_classifier_ensemble_embeddings.joblib" }
```

Remove or change this file to switch models. If the file is absent or the path does not exist,
the service falls back to rule-based classification.
