# Prompt Difficulty Classification Service

MVP FastAPI service for classifying user prompts into two routing labels:

- `cheap_ok`
- `escalate`

Because there is no human-labeled dataset yet, the first stage uses rule-based pseudo-labeling.
Then a lightweight classifier can be trained on the pseudo-labeled dataset.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload --port 8080
```

## Label a dataset from CLI

```bash
python3 scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

## Train a model from CLI

```bash
python3 scripts/train_model.py --input data/report_labeled_binary.json --model-output models/prompt_classifier.joblib
```

## Classify a prompt

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor this authentication flow and explain the tradeoffs"}'
```

## Main API endpoints

- `GET /health`
- `POST /classify`
- `POST /label-dataset`
- `POST /train`
