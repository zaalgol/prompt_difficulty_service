# CLAUDE.md

## Project

FastAPI MVP service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

This is not production yet. The current dataset is only Claude Code prompt history from building the project, so labels are pseudo-labels, not human ground truth.

## Core rule

When unsure, classify as `escalate`.

The main failure to avoid is sending a difficult prompt to a weak model.

## Current workflow

Use the project skills for detailed steps:

- `.claude/skills/windows-python-setup.md`
- `.claude/skills/dataset-labeling.md`
- `.claude/skills/training-and-inference.md`
- `.claude/skills/implementation-guidelines.md`

## Running locally

Always **activate the venv first** — `uvicorn`/`pytest`/`python` live inside it, so
running them without activation gives "not recognized" (PowerShell) or "command
not found" (macOS/Linux). This is the most common new-user stumble.

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1` then
  `uvicorn app.main:app --reload --port 8081`
- macOS/Linux: `source .venv/bin/activate` then
  `uvicorn app.main:app --reload --port 8081`

The prompt shows `(.venv)` when active. To skip activation, call the venv binary
directly: `.\.venv\Scripts\uvicorn.exe ...` (Windows) or `.venv/bin/uvicorn ...`
(macOS/Linux). See README "Run API" and "Classify a prompt" for cross-platform
request examples. On Windows, prefer building request JSON with `ConvertTo-Json`
over hand-typed JSON strings to avoid brace/quote `JSON decode error`s.

If `pip install -r requirements.txt` fails on the `en_core_web_lg` line with a
GitHub `504 Gateway Time-out`, it is transient (pip downloads all wheels before
installing, so one flaky URL aborts the run) — not a broken environment. Retry, or
check `pip show en_core_web_lg`; if present, the model is already installed.

## Expected project structure

```text
app/        FastAPI app, schemas, labeling, dataset, modeling, logging
scripts/    CLI scripts for labeling, training, and prompt classification
data/       input and generated datasets
models/     trained model artifacts
logs/       rotating log files (gitignored)
```

Key files:

```text
app/config.py          paths, constants, LOG_LEVEL env var
app/logging_config.py  central logging setup (terminal + rotating file)
app/labeling.py        rule-based pseudo-labeler
app/dataset.py         JSON load/save/labeling
app/modeling.py        training and inference, all model variants
app/main.py            FastAPI endpoints
app/schemas.py         Pydantic models
app/ml.py              pure-Python ML utilities (TF-IDF, LogReg, etc.)
app/presidio_service/  PII anonymization for /anonymize (Presidio + Faker)
service_config.json    runtime override for model_path
```

## /anonymize endpoint

`POST /anonymize` takes a single prompt and returns it with PII replaced by
realistic fake values (Presidio for detection, Faker for replacement). Two
guarantees:

- Coherence: the same value maps to the same fake across prompts sharing a
  `session_id`. The per-session vault (original→fake map) has two backends, chosen
  by `vault_backend` in `service_config.json` (or `VAULT_BACKEND` env var):
  `"memory"` (default — in-process dict, single-process, lost on restart) or
  `"redis"` (`REDIS_URL`, with a per-session TTL, coherence across
  processes/restarts; a `session_id` request fails closed with 503 when Redis is
  down). Either way the original PII stays server-side and is never returned or
  logged; requests without a `session_id` never use the vault. Both vault classes
  live in `app/presidio_service/service.py` (`InMemorySessionVault`, `SessionVault`).
- Semantics: values the answer depends on are kept unchanged. The service
  auto-detects them with a keyword heuristic (`app/presidio_service/semantics.py`)
  — e.g. it keeps `DATE_TIME` when the prompt asks a retirement/age/countdown
  question — and the caller can also force types via `preserve_entity_types`.
  Auto-detection only ever adds preservation; disable it per request with
  `auto_preserve: false`.

Dataset pseudo-labeling is CLI-only (`scripts/label_dataset.py`); it is no longer
an HTTP endpoint.

Original dataset path:

```text
data/report.json
```

Generated labeled dataset:

```text
data/report_labeled_binary.json
```

Default training variant (no flag to `scripts/train_model.py`): **Embeddings +
LogReg**. `embedding_provider` in `service_config.json` selects `"local"`
(`nomic-ai/nomic-embed-text-v1.5` via SentenceTransformers) or `"openai"`
(`text-embedding-3-small`, requiring `OPENAI_API_KEY`). Provider and model settings
are serialized into the artifact, so changing them requires retraining. Use
`--use-tfidf` for the key-free TF-IDF + LogReg baseline. Every training run is
non-destructive and writes a UTC-timestamped artifact under `models/`.

Local models that set `embedding_trust_remote_code: true` must also pin
`embedding_model_revision`. Training may persist embeddings to the disk cache;
loading an artifact for API or CLI inference automatically switches that cache to
memory-only so served prompts are not written to disk.

Legacy TF-IDF + LogReg artifact path:

```text
models/prompt_classifier.joblib
```

Active model is set by `model_path` in `service_config.json`. The committed default
points at the key-free TF-IDF baseline (`models/prompt_classifier.joblib`), so a
fresh clone — where `models/` is gitignored and empty — runs rule-based until a
model is trained and `model_path` is updated. This keeps `pip install` + run
working with no API key. The CLI (`scripts/classify_prompt.py`) resolves the same
path via `app.config.resolve_model_path`, so CLI and API select the same artifact.

`run_classify_dummy_on_startup` and `run_anonymize_dummy_on_startup` independently
control synthetic startup requests; both default to true. They initialize the
complete classifier and Presidio/Faker paths without persisting dummy cache or
vault state. Setting a flag false restores lazy initialization for that endpoint.

## Labels

`cheap_ok` means the prompt appears simple enough for a cheaper/faster model.

`escalate` means the prompt likely needs stronger reasoning, more context, or safer handling.

Do not add more labels unless explicitly requested.
