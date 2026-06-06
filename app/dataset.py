import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.labeling import label_prompt_dict


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def label_dataset(input_path: str | Path, output_path: str | Path) -> Tuple[int, Dict[str, int]]:
    data = load_json(input_path)

    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Input JSON must contain a top-level 'prompts' array.")

    labeled_prompts = [label_prompt_dict(item) for item in prompts]

    label_counts = Counter(item["difficulty_label"] for item in labeled_prompts)

    output = dict(data)
    output["prompts"] = labeled_prompts
    output["labeling_summary"] = {
        "labeling_method": "rule_based_pseudo_label_mvp_v1",
        "total_prompts": len(labeled_prompts),
        "label_counts": dict(label_counts),
        "warning": (
            "These are pseudo-labels created by deterministic rules, not human ground truth. "
            "Use them only as an initial MVP dataset."
        ),
    }

    save_json(output, output_path)

    return len(labeled_prompts), dict(label_counts)
