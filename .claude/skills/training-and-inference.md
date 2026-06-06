# Skill: Training and Inference

Use this when training or running the FastAPI service.

## Train model

```bash
python scripts/train_model.py --input data/report_labeled_binary.json --model-output models/prompt_classifier.joblib
```

The first model should stay simple:

- TF-IDF
- LogisticRegression
- binary labels only

## Main metric

Prioritize low false cheap rate:

```text
hard/escalate prompt incorrectly predicted as cheap_ok
```

This is more important than general accuracy.

## Run API

```bash
uvicorn app.main:app --reload --port 8080
```

## Health check

```bash
curl http://localhost:8080/health
```

## Classify one prompt

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8080/classify" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Refactor the authentication flow and explain the security tradeoffs"}'
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

If no trained model exists, `/classify` should use rule-based fallback.
