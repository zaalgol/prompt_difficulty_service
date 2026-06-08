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

If `pip install` fails on the `en_core_web_lg` line with a GitHub `504 Gateway
Time-out`, that is a transient server-side error, not a broken setup — pip
downloads everything before installing, so one flaky URL aborts the whole run.
Just retry, or install the deps and the model separately so a model hiccup cannot
block the rest:

```bash
pip install -r requirements.txt --no-deps en_core_web_lg  # skip; install rest first
pip install -r requirements.txt                            # retry the model line
python -m spacy download en_core_web_lg                     # or pull via spaCy's own route
```

To confirm what is already installed, run `pip show en_core_web_lg` (or
`pip list`) — if it lists the package, the install succeeded earlier and the 504
is just a redundant re-download.

To use a smaller model instead, run `python -m spacy download en_core_web_sm` and
set `"presidio_nlp_model": "en_core_web_sm"` in `service_config.json`.

The `/anonymize` endpoint keeps a per-session consistency vault whose backend is
chosen by `vault_backend` in `service_config.json` (or the `VAULT_BACKEND` env
var): `"memory"` (default, in-process) or `"redis"`. Redis gives cross-process,
restart-surviving coherence — point the service at it via the `REDIS_URL` env var
or `redis_url` in `service_config.json` (default `redis://localhost:6379/0`).
`/classify` and one-shot `/anonymize` calls (no `session_id`) never use the vault.

## Run API

First **activate the venv** in the same terminal — without it, `uvicorn` is "not
recognized" (Windows) or "command not found" (macOS/Linux), because it only lives
inside the venv:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8081
```

Linux / macOS:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8081
```

The prompt shows `(.venv)` once it is active. To skip activation, call the venv's
binary directly — `.\.venv\Scripts\uvicorn.exe ...` (Windows) or
`.venv/bin/uvicorn ...` (macOS/Linux). The server is up when you see
`Application startup complete` and `Uvicorn running on http://127.0.0.1:8081`.

## Run tests

```bash
pytest
```

## Label a dataset from CLI

```bash
python scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

## Train a model from CLI

Default (Embeddings + LogReg — no flag needed):

```bash
python scripts/train_model.py --input data/report_labeled_binary.json
```

Active model (Embeddings + Ensemble — LGBM, XGBoost, CatBoost):

```bash
python scripts/train_model.py --input data/report_labeled_binary.json --use-ensemble-embeddings
```

Each training run **does not overwrite** the previous artifact: it writes a new
file into the `models/` folder with a UTC timestamp prefix on the name, e.g.
`models/2026-06-08T08-11-03Z__prompt_classifier_embeddings.joblib`. Pick the file
you want and point `service_config.json` (`model_path`) at it to make the service
load it.

Other variants (pass one flag):

| Flag | Model |
|------|-------|
| *(none)* | **Embeddings + LogReg (default)** |
| `--use-embeddings` | Embeddings + LogReg (same as default, explicit) |
| `--use-tfidf` | TF-IDF + LogReg (legacy baseline) |
| `--use-lgbm` | TF-IDF + LightGBM |
| `--use-lgbm-embeddings` | Embeddings + LightGBM |
| `--use-lgbm-embeddings-tuned` | Embeddings + LightGBM (RandomSearch) |
| `--use-lgbm-embeddings-optuna` | Embeddings + LightGBM (Optuna TPE) |
| `--use-ensemble-embeddings` | Embeddings + LGBM + XGBoost + CatBoost |
| `--compare` | Train all 7 variants and print a comparison table |

Embedding variants require `OPENAI_API_KEY` to be set. Because the default is now
an embeddings model, the no-flag command needs that key too; use `--use-tfidf` for
a key-free run.

## Classify a prompt

**Windows (PowerShell) — recommended.** Build the JSON with `ConvertTo-Json`
instead of hand-typing it. Hand-written JSON in single quotes is easy to break —
a stray `)` instead of `}`, or an apostrophe inside the prompt, yields a confusing
`JSON decode error` from the server. Letting PowerShell build and escape it avoids
all of that:

```powershell
$body = @{ prompt = "Refactor the authentication flow and explain the security tradeoffs" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8081/classify" -Method Post -ContentType "application/json" -Body $body
```

You can optionally pass `total_input_tokens` (the prompt's input-token count).
Models trained with it use it as an extra feature; omit it when unknown and it
maps to a neutral value, while models trained before the feature ignore it:

```powershell
$body = @{ prompt = "Refactor this auth flow"; total_input_tokens = 45000 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8081/classify" -Method Post -ContentType "application/json" -Body $body
```

If you prefer a one-liner with a literal JSON string, make sure it closes with `}`
(not `)`) and keep it on a single line:

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/classify" -Method Post -ContentType "application/json" -Body '{"prompt":"Refactor this auth flow","total_input_tokens":45000}'
```

**Linux / macOS (curl)** — `\` below is a bash line-continuation; on PowerShell use
the `Invoke-RestMethod` examples above instead:

```bash
curl -X POST http://localhost:8081/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor this authentication flow and explain the tradeoffs"}'
```

With the optional token count:

```bash
curl -X POST http://localhost:8081/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor this auth flow", "total_input_tokens": 45000}'
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
- The session vault has two backends, chosen by `vault_backend`:
  - **`memory`** (default): mappings live in an in-process dict — single-process
    and lost on restart, no external service. The number of sessions is
    LRU-bounded by `presidio_max_sessions`.
  - **`redis`**: one key per `session_id` holds the original→fake map, so
    coherence survives across worker processes and restarts. Each key carries a
    TTL refreshed on access (`presidio_session_ttl_seconds`). If Redis is
    unreachable, a `session_id` request fails closed with **503** rather than
    silently losing coherence.
  Either way the original PII stays server-side — never returned to callers or
  logged — and each entity type keeps at most `presidio_max_entries_per_type`
  mappings. Requests **without** a `session_id` never use the vault.
- `/health` reports anonymizer readiness under `anonymizer.engines_loaded` and the
  active backend under `anonymizer.vault_backend`. Engines load lazily on first
  `/anonymize`; set `"presidio_warm_on_startup": true` in `service_config.json` to
  load them at startup instead.

Tunable in `service_config.json`: `presidio_nlp_model`, `presidio_score_threshold`,
`vault_backend`, `presidio_max_sessions`, `redis_url`, `redis_socket_timeout`,
`presidio_session_ttl_seconds`, `presidio_max_entries_per_type`,
`presidio_warm_on_startup`.

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
  "model_path": "models/2026-06-08T08-11-03Z__prompt_classifier_embeddings.joblib",
  "log_level": "INFO"
}
```

Remove or change this file to switch models. If the file is absent or the path does not exist,
the service falls back to rule-based classification.
