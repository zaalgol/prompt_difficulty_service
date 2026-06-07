# Prompt Difficulty Classification Service

MVP FastAPI service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

Because there is no human-labeled dataset yet, the first stage uses rule-based pseudo-labeling.
Then a lightweight classifier can be trained on the pseudo-labeled dataset.

## Install

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload --port 8081
```

## Run tests

```bash
pytest
```

## Label a dataset from CLI

```bash
python scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

## Train a model from CLI

Default (Embeddings + LightGBM):

```bash
python scripts/train_model.py --input data/report_labeled_binary.json --use-lgbm-embeddings
```

Active model (Embeddings + Ensemble — LGBM, XGBoost, CatBoost):

```bash
python scripts/train_model.py --input data/report_labeled_binary.json --use-ensemble-embeddings
```

Each training run saves a new timestamped file (e.g. `models/2026-06-07T06-11-24Z__prompt_classifier_ensemble_embeddings.joblib`).
Update `service_config.json` to point the service at the new file.

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

Windows (PowerShell) — single line (backticks must be the last character on a
line, so the paste-safe form is to keep it on one line):

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/classify" -Method Post -ContentType "application/json" -Body '{"prompt":"Refactor the authentication flow and explain the security tradeoffs"}'
```

Linux / macOS (curl) — `\` is a bash line-continuation; in PowerShell use the
Invoke-RestMethod example above instead:

```bash
curl -X POST http://localhost:8081/classify -H "Content-Type: application/json" -d '{"prompt": "Refactor this authentication flow and explain the tradeoffs"}'
```

## Classify from CLI (no server needed)

```bash
python scripts/classify_prompt.py --prompt "design a scalable auth system"
```

## Main API endpoints

- `GET /health`
- `POST /classify`
- `POST /label-dataset`
- `POST /train`

## Logging

Logs are written to both the terminal and `logs/service.log` (rotating, 5 MB × 5 files).
Timestamps are in UTC (ISO-8601, e.g. `2026-06-07T06:11:24Z`) so logs are comparable
across containers regardless of host timezone.

The level can be set two ways, in order of precedence:

1. The `LOG_LEVEL` environment variable (highest priority, for per-deployment tuning).
2. The `log_level` key in `service_config.json` (committed default).
3. `INFO` if neither is set.

Windows (PowerShell):

```powershell
$env:LOG_LEVEL = "DEBUG"
uvicorn app.main:app --reload --port 8081
```

Linux / macOS:

```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload --port 8081
```

## Active model

The model loaded at startup is set in `service_config.json`:

```json
{
  "model_path": "models/prompt_classifier_ensemble_embeddings.joblib",
  "log_level": "INFO"
}
```

Remove or change this file to switch models. If the file is absent or the path does not exist,
the service falls back to rule-based classification.
