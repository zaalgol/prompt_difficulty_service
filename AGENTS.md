# AGENTS.md

## Project

FastAPI MVP service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

This is not production yet. The current dataset is only Codex prompt history from building the project, so labels are pseudo-labels, not human ground truth.

## Core rule

When unsure, classify as `escalate`.

The main failure to avoid is sending a difficult prompt to a weak model.

## Current workflow

Use the project skills for detailed steps:

- `.claude/skills/windows-python-setup.md`
- `.claude/skills/dataset-labeling.md`
- `.claude/skills/training-and-inference.md`
- `.claude/skills/implementation-guidelines.md`

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
service_config.json    runtime override for model_path
```

Original dataset path:

```text
data/report.json
```

Generated labeled dataset:

```text
data/report_labeled_binary.json
```

Default trained model (TF-IDF + LogReg baseline):

```text
models/prompt_classifier.joblib
```

Active model is set in `service_config.json` (currently the ensemble embeddings model).

## Labels

`cheap_ok` means the prompt appears simple enough for a cheaper/faster model.

`escalate` means the prompt likely needs stronger reasoning, more context, or safer handling.

Do not add more labels unless explicitly requested.
