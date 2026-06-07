# Skill: Implementation Guidelines

Use this when editing the code.

## Rules

1. Keep the code simple.
2. Keep labeling, dataset handling, model training, and API inference separated.
3. Do not introduce production assumptions.
4. Do not add external services unless requested.
5. Keep Windows PowerShell compatibility.
6. Do not change the label schema unless explicitly requested.
7. Do not start with 3–5 labels.
8. Do not present pseudo-labels as human ground truth.
9. After setup changes, provide exact commands.
10. Prefer conservative routing: uncertain means `escalate`.

## Current files

```text
app/main.py            FastAPI endpoints
app/schemas.py         Pydantic models
app/labeling.py        rule-based pseudo-labeler
app/dataset.py         JSON load/save/labeling
app/modeling.py        training and inference (all model variants)
app/ml.py              pure-Python ML utilities (TF-IDF, LogReg, etc.)
app/presidio_service/  PII anonymization for /anonymize (Presidio + Faker)
app/config.py          paths, constants, LOG_LEVEL
app/logging_config.py  central logging setup — use get_logger(__name__) in every module
scripts/               CLI wrappers
```

## Development priority

First verify the pipeline works end-to-end:

```text
report.json -> pseudo-labels -> train model -> API classify
```

Only then improve labeling rules or model quality.
