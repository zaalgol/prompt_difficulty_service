# Skill: Dataset Labeling

Use this when working on pseudo-labeling the original prompt dataset.

## Input

Original dataset:

```text
data/report.json
```

It must contain:

```json
{
  "prompts": [
    {
      "prompt": "..."
    }
  ]
}
```

## Output

Generated dataset:

```text
data/report_labeled_binary.json
```

Each prompt item should include:

```json
{
  "difficulty_label": "cheap_ok | escalate",
  "difficulty_confidence": 0.0,
  "labeling_reason": "...",
  "labeling_method": "rule_based_pseudo_label_mvp_v1",
  "labeling_features": {}
}
```

## Command

Windows (PowerShell) — **activate venv first**:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

Linux / macOS:

```bash
source .venv/bin/activate
python scripts/label_dataset.py --input data/report.json --output data/report_labeled_binary.json
```

## Labeling policy

Use only two labels:

- `cheap_ok`
- `escalate`

When unsure, use `escalate`.

The labels are pseudo-labels, not ground truth.
