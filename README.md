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

The `/anonymize` endpoint uses a spaCy English model for Presidio's NER. It is
pinned in `requirements.txt`, so the `pip install` above already includes it — no
separate `spacy download` step is needed.

To use a smaller model instead, run `python -m spacy download en_core_web_sm` and
set `"presidio_nlp_model": "en_core_web_sm"` in `service_config.json`.

The `/anonymize` endpoint stores its per-session consistency vault in Redis. For
cross-request coherence (requests sharing a `session_id`), point the service at a
Redis instance via the `REDIS_URL` env var or `redis_url` in `service_config.json`
(default `redis://localhost:6379/0`). `/classify` and one-shot `/anonymize` calls
(no `session_id`) do not need Redis. The test suite uses an in-process `fakeredis`,
so no server is required to run `pytest`.

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

You can optionally pass `total_input_tokens` (the prompt's input-token count).
Models trained with it use it as an extra feature; omit it when unknown and it
maps to a neutral value, while models trained before the feature ignore it:

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/classify" -Method Post -ContentType "application/json" -Body '{"prompt":"Refactor this auth flow","total_input_tokens":45000}'
```

Linux / macOS (curl) — `\` is a bash line-continuation; in PowerShell use the
Invoke-RestMethod example above instead:

```bash
curl -X POST http://localhost:8081/classify -H "Content-Type: application/json" -d '{"prompt": "Refactor this authentication flow and explain the tradeoffs"}'
```

You can optionally pass `total_input_tokens` (the prompt's input-token count).
Models trained with it use it as an extra feature; omit it when unknown and it
maps to a neutral value, while models trained before the feature ignore it:

```bash
curl -X POST http://localhost:8081/classify -H "Content-Type: application/json" -d '{"prompt": "Refactor this auth flow", "total_input_tokens": 45000}'
```

## Classify from CLI (no server needed)

```bash
python scripts/classify_prompt.py --prompt "design a scalable auth system" --total-input-tokens 45000
```

## Main API endpoints

- `GET /health`
- `POST /classify`
- `POST /anonymize` — detect PII in a prompt and replace it with realistic, fake
  values before forwarding it to an LLM. Replacements are consistent across prompts
  that share a `session_id`. Values the answer depends on are kept unchanged so the
  result stays correct: the service auto-detects them (e.g. it keeps a date of birth
  when the prompt asks about retirement/age) and the caller can also force types via
  `preserve_entity_types`. Set `"auto_preserve": false` to disable auto-detection.
  Requires Presidio plus a spaCy model — see Install.
- `POST /train`

Dataset pseudo-labeling is now a CLI step (`scripts/label_dataset.py`), not an HTTP
endpoint.

Anonymizer notes:

- Replacements for addressable types use **reserved, non-routable** values
  (`example.com`, `192.0.2.0/24` documentation IPs, `(555) 555-01xx` numbers) so the
  output can never point a model or agent at a real host, mailbox, or phone line.
- Session mappings are stored in **Redis** (one key per `session_id`, holding the
  original→fake map), so coherence survives across worker processes and restarts.
  The original PII lives only in Redis — never returned to callers or logged. Each
  session key carries a TTL refreshed on access (`presidio_session_ttl_seconds`),
  and each entity type keeps at most `presidio_max_entries_per_type` mappings.
  Requests **without** a `session_id` never touch Redis (intra-prompt coherence only).
- If Redis is unreachable, a request **with** a `session_id` fails closed with
  **503** rather than silently losing coherence; `/health` reports the backend under
  `anonymizer.redis_connected`. Configure the connection with the `REDIS_URL` env var
  (or `redis_url` in `service_config.json`); default `redis://localhost:6379/0`.
- `/health` reports anonymizer readiness under `anonymizer.engines_loaded`. Engines
  load lazily on first `/anonymize`; set `"presidio_warm_on_startup": true` in
  `service_config.json` to load them at startup instead.

Tunable in `service_config.json`: `presidio_nlp_model`, `presidio_score_threshold`,
`redis_url`, `redis_socket_timeout`, `presidio_session_ttl_seconds`,
`presidio_max_entries_per_type`, `presidio_warm_on_startup`.

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
