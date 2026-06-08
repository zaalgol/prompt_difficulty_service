# Documentation Review

Date: 2026-06-08

## Scope

This is a static documentation review for a developer who does not know the
codebase and wants to:

1. Install it on Windows, Linux, or macOS.
2. Run and verify the API.
3. Use classification and anonymization.
4. Train or select a model.
5. Make code changes without breaking important behavior.

Reviewed:

- `README.md`
- `CLAUDE.md`
- `.claude/skills/windows-python-setup.md`
- `.claude/skills/dataset-labeling.md`
- `.claude/skills/training-and-inference.md`
- `.claude/skills/implementation-guidelines.md`
- Relevant implementation and deployment files used to verify the docs

No application code or tests were run. No files other than this review were
changed.

## Overall Assessment

The documentation is useful for a developer who already has the author's local
data and model artifacts, especially on Windows. It is not yet a reliable
fresh-clone guide.

Approximate onboarding quality:

- Windows local setup: 7/10
- Linux local setup: 5/10
- macOS local setup: 5/10
- Docker setup: 2/10
- API usage: 6/10
- Contributor/change guidance: 5/10

The strongest parts are the explicit virtual-environment commands, conservative
routing rule, training-variant table, PowerShell request guidance, and detailed
anonymizer behavior notes.

The largest problems are factual or operational:

- An installation recovery command does not do what its comment claims.
- Documented datasets and models are gitignored and absent from a fresh clone.
- `service_config.json` points to an untracked timestamped model.
- The README calls the ensemble the active model, but the configured model is
  Embeddings + LogReg.
- Embedding-based inference needs API access too, not only training.
- The CLI classifier and HTTP service do not select models the same way.
- Docker files exist, but there is no Docker documentation.

## High-Priority Findings

### 1. The spaCy model recovery command is incorrect

Evidence:

- `README.md:40` recommends:
  `pip install -r requirements.txt --no-deps en_core_web_lg`
- Its comment says this will skip the model and install the remaining
  dependencies.

That command does not remove the `en_core_web_lg` requirement from
`requirements.txt`. It still processes the full requirements file, and it also
adds `en_core_web_lg` as a separate install target. `--no-deps` changes dependency
resolution; it does not exclude one line from a requirements file.

Impact:

A developer following the documented recovery path can hit the same failed URL
again or receive another confusing package-resolution error.

Recommended change:

- Remove the incorrect command.
- Give one tested recovery path, such as installing from a temporary requirements
  file with the model line excluded, then installing the model separately.
- Provide PowerShell and bash variants because filtering a file differs by shell.
- Avoid asserting that `python -m spacy download` is a different route unless
  that has been verified for the pinned model/version.

### 2. Fresh-clone behavior is not explained

Evidence:

- `.gitignore:7-9` ignores all of `data`, `models`, and `logs`.
- `README.md:90-107` immediately uses:
  - `data/report.json`
  - `data/report_labeled_binary.json`
  - model output under `models/`
- `CLAUDE.md:99-121` presents these as standard project paths.
- `service_config.json:2` points to
  `models/2026-06-08T08-11-03Z__prompt_classifier_embeddings.joblib`.
- The configured model is not tracked.

Impact:

On a fresh clone:

- The labeling command has no input dataset.
- The training command has no labeled dataset.
- The configured model path does not exist.
- The API starts in rule-based fallback mode.
- `/health` reports `ready: false` even though requests can still be served by
  the fallback classifier.

None of those outcomes is necessarily wrong, but the current docs make them look
like setup failures.

Recommended change:

Add a "Fresh clone expectations" section near the top of the README:

- State that datasets and model artifacts are intentionally not committed.
- Explain where a developer should obtain or place `report.json`.
- State that the first startup uses rule-based fallback.
- Show the expected `/health` response in fallback mode.
- Explain how to train a key-free TF-IDF model, update `model_path`, restart, and
  verify `model_loaded: true`.
- Consider committing empty `data/`, `models/`, and `logs/` directories via
  placeholder files, or explicitly tell the user that the application creates
  output directories when needed.

### 3. The README contradicts the active configuration

Evidence:

- `README.md:104` labels Embeddings + Ensemble as the "Active model."
- `README.md:260` and `service_config.json:2` point to
  `prompt_classifier_embeddings.joblib`, which is the Embeddings + LogReg
  variant.
- `CLAUDE.md:124` correctly says the current model is Embeddings + LogReg.

Impact:

A developer cannot tell which training command reproduces the configured model.
This also creates avoidable drift between README and Claude guidance.

Recommended change:

- Rename the README subsection at line 104 to "Ensemble variant" unless it is
  actually active.
- Prefer describing model selection generically instead of embedding a dated
  artifact filename in several files.
- Keep the current active model in one authoritative location:
  `service_config.json`.

### 4. Embedding-based runtime requirements are missing

Evidence:

- `README.md:130-132` says embedding variants require `OPENAI_API_KEY` while
  discussing training.
- `.claude/skills/training-and-inference.md:31-32` does the same.
- `app/ml.py` uses the embedding API from `EmbeddingVectorizer.transform()`, so a
  new prompt can require network access and credentials during inference.
- `app/modeling.py` catches inference failure and returns an `escalate` result
  with `method: "fail_closed"`.

Impact:

A developer can successfully load an embedding artifact, believe the trained
model is operating, and receive only fail-closed escalation because the API key
or network is unavailable. The service does not necessarily return an obvious
HTTP error.

Recommended change:

Document runtime requirements separately from training requirements:

- How to set `OPENAI_API_KEY` on PowerShell and bash/zsh.
- That embedding inference can make outbound API calls for uncached prompts.
- That missing credentials/network access returns a conservative
  `method: "fail_closed"` result.
- That TF-IDF is the fully local option.
- Whether embeddings or prompt-derived data are sent to an external provider.

### 5. CLI and HTTP classification model selection differs

Evidence:

- The HTTP service reads `model_path` from `service_config.json`.
- `scripts/classify_prompt.py` calls `classify_with_model()` without a path.
- `classify_with_model()` uses the legacy default
  `models/prompt_classifier.joblib`, not the configured active model.
- `README.md:180-184` presents the CLI as a serverless equivalent without warning
  about this difference.

Impact:

The same prompt can be classified by different models, or by trained-model versus
fallback mode, depending on whether the developer uses the CLI or API.

Recommended change:

- Explicitly document the current difference.
- Prefer changing the CLI later to use `service_config.json` by default and offer
  a `--model-path` override.
- Once code behavior is aligned, document one model-selection rule for both
  interfaces.

### 6. Docker is present but effectively undocumented

Evidence:

- The repository contains `Dockerfile` and `docker-compose.yml`.
- The README contains no Docker build, run, health-check, volume, environment, or
  troubleshooting instructions.
- `docker-compose.yml` mounts `data/` and `models/`, which are empty on a fresh
  clone.
- The image copies `service_config.json`, which references an untracked model.
- Compose does not define Redis or pass `OPENAI_API_KEY`, `VAULT_BACKEND`,
  `REDIS_URL`, or `LOG_LEVEL`.

Impact:

Docker looks supported but a newcomer cannot know the intended mode. A fresh
Compose startup uses rule-based fallback, embedding inference lacks credentials,
and Redis-backed session coherence requires manual configuration.

Recommended change:

Add a Docker section with:

- `docker compose up --build`
- Expected fallback behavior on first startup
- `curl`/PowerShell health checks
- How host `data/` and `models/` directories are used
- How to pass environment variables without committing secrets
- An optional Redis Compose profile or a documented external Redis setup
- How logs are persisted, if persistence is intended
- Linux/macOS/Windows notes for bind mounts

## Medium-Priority Findings

### 7. The primary README lacks an anonymize request and response example

`README.md:186-227` describes `/anonymize` in detail but does not show a complete
request or response. The only request example is hidden in the Claude training
skill, which ordinary developers may never read.

Add examples for PowerShell and curl covering:

- One-shot anonymization
- Two requests with the same `session_id`
- `preserve_entity_types`
- `auto_preserve: false`
- Response fields and entity offsets
- 422, 500, and 503 behavior

The same-session example is especially important because coherence is a central
feature and cannot be understood from a single request.

### 8. Cross-platform prerequisites are incomplete

The README assumes `py -3.12` or `python3.12` already exists. It does not state:

- Supported Python versions in the primary README
- How to verify the selected interpreter
- That some Linux distributions require a separate venv package
- What to do when `python3.12` is not the executable name
- The Windows PowerShell execution-policy failure and the existing
  no-activation alternative
- Whether Apple Silicon is supported
- Expected disk/download cost of the large spaCy model and ML dependencies

The Windows skill allows Python 3.13 while the README documents only 3.12. The
phrase "current locked dependencies" in
`.claude/skills/windows-python-setup.md:9` is also misleading because most
dependencies are not version-pinned and there is no lock file.

Recommended change:

Create one support matrix in the README and have Claude guidance link to it:

| Platform | Supported Python | Setup command | Known caveats |
|----------|------------------|---------------|---------------|
| Windows PowerShell | 3.12, verified alternatives | ... | execution policy |
| Linux | ... | ... | venv package |
| macOS Intel/Apple Silicon | ... | ... | wheel availability |

Only claim versions/platforms that CI or a documented manual check verifies.

### 9. Test documentation is too thin

`README.md:84-88` contains only `pytest`.

The suite includes:

- Unit and API tests
- Real Presidio/spaCy tests
- End-to-end tests that start servers on ports 8765 and 8766
- Tests whose behavior can depend on whether the configured untracked model is
  present

Recommended change:

Document:

- The canonical full command: `python -m pytest`
- Focused commands for classification and anonymization
- Expected skips when optional components are absent, if any remain optional
- Local ports used by end-to-end tests
- Whether tests make external embedding calls
- How to force deterministic key-free/fallback test behavior
- Approximate successful test count and duration, marked as illustrative rather
  than guaranteed

### 10. Contributor guidance lacks an architecture and change map

`CLAUDE.md` and `.claude/skills/implementation-guidelines.md` list files, but a
new contributor still has to infer the runtime flows.

Missing flows:

- Startup and model loading
- `/classify` trained, fallback, confidence override, and fail-closed paths
- `/anonymize` detection, preservation, replacement, and session-vault paths
- In-memory versus Redis ownership and lifecycle
- Dataset labeling to training to artifact activation
- Which configuration is read at import time versus startup time

Recommended change:

Add a compact architecture section or Mermaid diagram, plus a "change map":

- Adding a request field: schema, endpoint, model/service, tests, docs
- Adding an anonymized entity type: recognizer, precedence/protection,
  replacement operator, session consistency, adversarial tests
- Changing a configuration key: config parser, service behavior, sample config,
  README table, tests
- Changing model behavior: false-cheap tests, artifact compatibility, activation

### 11. Security-sensitive contributor rules are incomplete

The anonymizer handles PII, credentials, and private keys, but the general Claude
implementation guidance does not make its security invariants prominent.

Recommended additions:

- Never log prompts, entity values, secrets, vault mappings, or exception text
  that may contain them.
- Preserve structured syntax when replacing values.
- Test both leak prevention and semantic preservation.
- Test repeated values, overlapping recognizers, multiple requests in one
  session, cross-session isolation, Redis failure, and eviction.
- Use only synthetic credentials in tests and documentation.
- Treat anonymization false negatives as security failures and over-redaction as
  correctness failures.

### 12. Configuration is scattered instead of specified

The README lists tunable names at `README.md:224-227`, but does not provide types,
defaults, allowed values, environment overrides, startup requirements, or
reload behavior.

Add a configuration table covering:

- `model_path`
- `log_level` / `LOG_LEVEL`
- `min_cheap_confidence`
- `embedding_model`
- `presidio_nlp_model`
- `presidio_score_threshold`
- `vault_backend` / `VAULT_BACKEND`
- `redis_url` / `REDIS_URL`
- `redis_socket_timeout`
- `presidio_max_sessions`
- `presidio_session_ttl_seconds`
- `presidio_max_entries_per_type`
- `presidio_warm_on_startup`

Also explain that most values are loaded when the process imports/starts and
therefore require a restart.

### 13. `/health` semantics need a concrete explanation

The README mentions individual anonymizer fields but not the complete meaning of
the response.

Important distinctions:

- `status: "ok"` is process liveness.
- `ready` currently reflects whether a trained classifier artifact loaded.
- Rule-based fallback can serve `/classify` while `ready` is false.
- `anonymizer.engines_loaded` is false before the first anonymization request
  when lazy loading is enabled.
- Redis is deliberately not probed by `/health`.

Without this explanation, a fresh-clone developer may diagnose normal fallback
or lazy-loading behavior as a failed installation.

### 14. `/train` is listed without its contract or its behavioral difference

`README.md:197` lists `POST /train` but provides no request, response, security
warning, or model-variant explanation.

The endpoint calls the TF-IDF training path, while the CLI defaults to embeddings.
That difference should be documented. The docs should also say whether this
endpoint is intended only for local development, because it accepts filesystem
paths and performs expensive work synchronously.

### 15. Dependency reproducibility is overstated

Only the Presidio/Faker/spaCy subset is pinned. Core packages such as FastAPI,
Pydantic, scikit-learn, LightGBM, XGBoost, CatBoost, pytest, Redis, and httpx are
unbounded.

Impact:

Two developers can follow identical installation instructions and receive
different environments. This weakens all platform-support claims and can change
serialized-model compatibility.

Recommended change:

- Stop referring to the dependency set as locked.
- Document the known-good Python version and dependency snapshot.
- Consider a lock/constraints file later, especially for model artifact
  compatibility and cross-platform setup.

## Documentation Structure Recommendation

The README would be easier for a newcomer if reordered as:

1. What the service does
2. Fresh-clone behavior and prerequisites
3. Quick start using local rule-based fallback
4. Verify with `/health`, `/classify`, and `/anonymize`
5. Configuration and environment variables
6. Model/data setup and training
7. Redis-backed session coherence
8. Docker
9. Tests
10. Architecture and contributor workflow
11. Troubleshooting
12. MVP limitations and security notes

Keep `CLAUDE.md` short and agent-focused:

- Core invariants
- Architecture summary
- Change/test map
- Security rules
- Links to detailed `.claude/skills/` instructions

Avoid duplicating volatile details such as the timestamped active model across
README, CLAUDE, and skills. Duplication has already produced a contradiction.

## What Is Already Good

- The two routing labels and conservative escalation rule are clear.
- The docs correctly disclose pseudo-labeling and lack of human ground truth.
- Windows PowerShell setup and request examples receive useful attention.
- The direct virtual-environment binary alternative is practical.
- The training-variant table closely matches the CLI flags.
- Non-destructive timestamped model output is explained.
- Anonymizer memory/Redis coherence behavior is described carefully.
- Redis failure is documented as fail-closed for session requests.
- Lazy anonymizer loading and its health field are mentioned.
- Logging location, rotation, UTC timestamps, and level precedence are clear.
- The Claude files are broken into focused skills instead of one oversized file.

## Suggested Documentation Acceptance Checklist

A documentation revision should be considered complete when a developer can,
using only tracked files:

- Identify a supported Python version on each platform.
- Create an environment and install dependencies with commands that are known to
  work.
- Understand what is intentionally absent from a fresh clone.
- Start the API without a model and recognize fallback mode as expected.
- Send successful classification and anonymization requests.
- Demonstrate same-session anonymization coherence.
- Understand when prompts cause external embedding API calls.
- Select a local TF-IDF or embedding model deliberately.
- Make CLI and HTTP classification use the intended artifact.
- Run focused and full tests without hidden port or credential surprises.
- Start the service with Docker and understand its volumes and environment.
- Modify classification or anonymization with a clear map of required tests and
  documentation updates.
