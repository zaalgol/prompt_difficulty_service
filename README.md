# Prompt Difficulty Classification Service

MVP FastAPI service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

Because there is no human-labeled dataset yet, the first stage uses rule-based pseudo-labeling.
Then a lightweight classifier can be trained on the pseudo-labeled dataset.

## What works out of the box

After `pip install -r requirements.txt` the service runs with **no API key, no
dataset, and no model file**:

- `POST /classify` answers using the built-in **rule-based classifier** (the safe
  fallback — "when unsure, escalate").
- `POST /anonymize` works using the bundled spaCy model.
- `GET /health` returns `status: "ok"` with `ready: false` and
  `mode: "rule_based_fallback"`. On a fresh clone that is **expected**, not an
  install error — `ready` only turns true once a trained model artifact loads.

`data/`, `models/`, and `logs/` are gitignored, so datasets and trained model
artifacts are **not** part of a clone. Training a model is an optional upgrade
(see [Train a model](#train-a-model-from-cli)): the key-free TF-IDF variant needs
nothing extra, while the embedding variants additionally require `OPENAI_API_KEY`
at runtime (see [Active model](#active-model)).

## Install

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `/anonymize` endpoint uses a spaCy English model (`en_core_web_lg`) for
Presidio's NER. It is pinned in `requirements.txt`, so the `pip install` above
already pulls it — there is **no** separate `spacy download` step.

If `pip install` fails on the `en_core_web_lg` line with a GitHub `504 Gateway
Time-out`, that is a transient server-side error, not a broken setup — pip
downloads every wheel before installing any, so one flaky URL aborts the whole
run. Just re-run the same command:

```bash
pip install -r requirements.txt   # resumes from cache; only re-fetches what's missing
```

To confirm the model is already installed, run `pip show en_core_web_lg` — if it
is listed, an earlier run succeeded and the 504 was just a redundant re-download.

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

Ensemble variant (Embeddings + LGBM, XGBoost, CatBoost):

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

Embedding variants require `OPENAI_API_KEY` to be set. Because the default training
variant is an embeddings model, the no-flag command needs that key too; use
`--use-tfidf` for a key-free run.

`OPENAI_API_KEY` is needed both to **train** an embedding variant **and to run the
service on one** — embedding inference calls the OpenAI API for each new prompt. If
the key or network is missing at request time, `/classify` fails closed and returns
`escalate` with `method: "fail_closed"` (no HTTP error is raised). The TF-IDF
variant (`--use-tfidf`) is fully local and never calls out.

Set the key with `$env:OPENAI_API_KEY = "sk-..."` (PowerShell) or
`export OPENAI_API_KEY=sk-...` (bash/zsh) in the same terminal before training or
starting the server.

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

This uses the **same model selection as the API**: `model_path` from
`service_config.json`, falling back to the key-free TF-IDF baseline, then to
rule-based when no artifact is present. Override the artifact with
`--model-path <file>`.

## Anonymize a prompt

`POST /anonymize` detects PII in a prompt and replaces it with realistic **fake**
values before you forward the prompt to an LLM, so the real data never leaves your
side. It gives two guarantees:

- **Coherence** — the same original value maps to the same fake across every prompt
  sharing a `session_id`, so a multi-turn conversation stays consistent.
- **Semantics** — values the answer depends on are kept unchanged, so the model
  still computes the right result.

### Request fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `prompt` | string (1–20000) | — | The text to anonymize. Required. |
| `session_id` | string (≤256) | `null` | Scopes coherence: same value → same fake within and across prompts sharing this id. Omit for one-shot. |
| `preserve_entity_types` | string[] (≤32) | `[]` | Entity types to keep **unchanged** (e.g. `["DATE_TIME"]`). Each must be UPPER_SNAKE. |
| `auto_preserve` | bool | `true` | Also auto-detect values the answer depends on (e.g. a birth date in a retirement question) and keep them. `false` = preserve only `preserve_entity_types`. |

### Response fields

| Field | Meaning |
|-------|---------|
| `anonymized_prompt` | The prompt with PII replaced (or preserved). |
| `session_id` | Echoes the request `session_id` (or `null`). |
| `entities` | One entry per detected entity: `entity_type`, character `start`/`end` in `anonymized_prompt`, and `action` (`anonymized` or `preserved`). |
| `preserved_entity_types` | Entity types kept unchanged for this request. |

The **original PII values are never returned or logged** — they stay server-side.

### One-shot

Windows (PowerShell):

```powershell
$body = @{ prompt = "Email me at john@acme.com about the Q3 report" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8081/anonymize" -Method Post -ContentType "application/json" -Body $body
```

Linux / macOS (curl):

```bash
curl -X POST http://localhost:8081/anonymize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Email me at john@acme.com about the Q3 report"}'
```

Response (fake values are random — only their *consistency* is guaranteed):

```json
{
  "anonymized_prompt": "Email me at popejoyce@example.com about the Q3 report",
  "session_id": null,
  "entities": [
    {"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 33, "action": "anonymized"}
  ],
  "preserved_entity_types": []
}
```

### Coherence across a session

Pass the same `session_id` on every related request; a repeated value gets the same
fake each time:

Windows (PowerShell):

```powershell
$body = @{ prompt = "Email alice@corp.com to confirm"; session_id = "thread-42" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8081/anonymize" -Method Post -ContentType "application/json" -Body $body
# -> "... harriskimberly@example.com to confirm"

$body = @{ prompt = "Did alice@corp.com reply yet?"; session_id = "thread-42" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8081/anonymize" -Method Post -ContentType "application/json" -Body $body
# -> "Did harriskimberly@example.com reply yet?"   (same fake as the first request)
```

Linux / macOS (curl):

```bash
curl -s -X POST http://localhost:8081/anonymize -H "Content-Type: application/json" \
  -d '{"prompt": "Email alice@corp.com to confirm", "session_id": "thread-42"}'
# -> "... harriskimberly@example.com to confirm"

curl -s -X POST http://localhost:8081/anonymize -H "Content-Type: application/json" \
  -d '{"prompt": "Did alice@corp.com reply yet?", "session_id": "thread-42"}'
# -> "Did harriskimberly@example.com reply yet?"   (same fake as the first request)
```

Requests **without** a `session_id` never touch the vault — each is independent.

### Keeping values the answer needs (semantics)

By default the service keeps data the answer depends on. A retirement question keeps
the birth date, so the model can still compute the answer:

```bash
curl -X POST http://localhost:8081/anonymize -H "Content-Type: application/json" \
  -d '{"prompt": "I was born on 1970-04-12, when can I retire?"}'
# date PRESERVED: "... born on 1970-04-12 ...", preserved_entity_types: ["DATE_TIME"]
```

Force-keep specific types yourself with `preserve_entity_types`, or turn the
auto-detection off with `auto_preserve: false` (then the date is anonymized to a
random one):

```bash
curl -X POST http://localhost:8081/anonymize -H "Content-Type: application/json" \
  -d '{"prompt": "I was born on 1970-04-12, when can I retire?", "auto_preserve": false}'
# date ANONYMIZED: "... born on 2018-01-18 ..."
```

### What gets detected and how it is replaced

Standard Presidio entity types are replaced with a same-shape fake. Network-/system-
addressable types use **reserved, non-routable** values, so the output can never
point a model (or an agent acting on its output) at a real host, mailbox, or phone:

| Entity type | Replacement |
|-------------|-------------|
| `PERSON`, `LOCATION`, `NRP`, `ORGANIZATION` | Faker name / city / country / company |
| `EMAIL_ADDRESS` | `<user>@example.com` (RFC 2606) |
| `URL` | `https://example.com/...` |
| `IP_ADDRESS` | `192.0.2.x` (RFC 5737 documentation range) |
| `PHONE_NUMBER` | `(555) 555-01xx` (fictional range) |
| `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_BANK_NUMBER`, `US_DRIVER_LICENSE` | Faker value of the same shape |
| `DATE_TIME` | Random `YYYY-MM-DD` (unless preserved) |

On top of Presidio's NER, a custom recognizer catches **developer secrets** —
private keys, AWS/OpenAI/Anthropic/GitHub/GitLab/Slack/Stripe/Google keys, JWTs,
`Authorization`/`Bearer` headers, session cookies, and credentialed connection URIs
(`postgres://user:pass@host`). It also recognizes contextual natural-language
forms such as `The API key is '...'`, `The password was ...`, and
`The secret is '...'`. The secret value is replaced with an opaque tag while the
surrounding syntax is preserved, so only the secret itself is removed:

```bash
curl -X POST http://localhost:8081/anonymize -H "Content-Type: application/json" \
  -d '{"prompt": "export OPENAI_API_KEY=sk-proj-abcdef...123456"}'
# -> "export OPENAI_API_KEY=<API_KEY>"
```

### Errors

| Status | When |
|--------|------|
| `422` | Request fails validation (empty prompt, prompt > 20000 chars, malformed `preserve_entity_types`). |
| `503` | Anonymization unavailable: Presidio/spaCy not installed or the NLP model missing, **or** the Redis vault backend is unreachable for a `session_id` request (fails closed so coherence is never silently lost). |
| `500` | Any other anonymization failure. Messages are deliberately generic — prompt text is never echoed back or logged. |

## Main API endpoints

- `GET /health`
- `POST /classify`
- `POST /anonymize` — detect PII in a prompt and replace it with realistic, fake
  values (Presidio + Faker). See [Anonymize a prompt](#anonymize-a-prompt) for
  request/response fields and examples. Requires Presidio plus a spaCy model — see
  Install.
- `POST /train` — train the TF-IDF model from a labeled dataset on the server
  (synchronous, local-dev convenience). Note it always trains the **TF-IDF** variant,
  while `scripts/train_model.py` defaults to embeddings; for anything beyond the
  baseline, prefer the CLI.

Dataset pseudo-labeling is now a CLI step (`scripts/label_dataset.py`), not an HTTP
endpoint.

Anonymizer vault and readiness notes:

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
On Windows, if another reload worker or service process holds that file during
rotation, the process that loses the rollover race continues in
`logs/service.<pid>.log` instead of interrupting logging with `WinError 32`.
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

The model loaded at startup is set by `model_path` in `service_config.json`:

```json
{
  "model_path": "models/prompt_classifier.joblib",
  "log_level": "INFO",
  "min_cheap_confidence": 0.8,
  "vault_backend": "memory"
}
```

The committed default points at the key-free TF-IDF baseline path. On a fresh clone
no artifact exists there, so the service starts in rule-based fallback (see
[What works out of the box](#what-works-out-of-the-box)) — this needs no API key.

To activate a trained model, train one and set `model_path` to its file (training
writes a timestamped artifact under `models/`; point `model_path` at that filename).
An embedding artifact additionally needs `OPENAI_API_KEY` at runtime. If the path is
empty, absent, or missing on disk, the service falls back to rule-based
classification.

## Docker

```bash
docker compose up --build
```

This builds the image and serves the API on `http://localhost:8081`. Host `./data`
and `./models` are bind-mounted into the container, so artifacts you train locally
are visible inside it. With no model artifact present, the container starts in
rule-based fallback, exactly like a local run.

The image does not set `OPENAI_API_KEY` or configure Redis. To run an embedding
model or the Redis vault backend, pass the relevant variables through — add them
under `environment:` in `docker-compose.yml` (or an `.env` file). Do not commit
secrets.
