import re
from typing import Any, Dict, Tuple

from app.config import LABEL_CHEAP_OK, LABEL_ESCALATE


COMPLEX_KEYWORDS = [
    "architecture",
    "design",
    "scale",
    "scaling",
    "distributed",
    "microservice",
    "security",
    "authentication",
    "authorization",
    "token",
    "jwt",
    "oauth",
    "vulnerability",
    "risk",
    "tradeoff",
    "trade-off",
    "database",
    "postgres",
    "redis",
    "kubernetes",
    "docker",
    "aws",
    "terraform",
    "ci/cd",
    "gitlab",
    "pipeline",
    "refactor",
    "debug",
    "bug",
    "error",
    "exception",
    "failing",
    "production",
    "deploy",
    "deployment",
    "performance",
    "latency",
    "queue",
    "scheduler",
    "worker",
    "cron",
    "thread",
    "async",
    "concurrency",
    "multi-step",
    "plan",
    "strategy",
    "pricing",
    "business",
    "cost",
    "model",
    "training",
    "classifier",
    "inference",
    "llm",
    "rag",
]

SIMPLE_KEYWORDS = [
    "translate",
    "rewrite",
    "rephrase",
    "format",
    "fix typo",
    "grammar",
    "summarize",
    "summary",
    "yaml syntax",
    "new line",
    "yes",
    "no",
    "ok",
    "do it",
    "try again",
]

APPROVAL_WORDS = {"yes", "ok", "okay", "do it", "try again", "go ahead", "run it"}


def count_words_by_whitespace(text: str) -> int:
    return len(text.split())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_features(prompt: str) -> Dict[str, Any]:
    normalized = normalize_text(prompt)
    word_count = count_words_by_whitespace(prompt)

    complex_hits = [kw for kw in COMPLEX_KEYWORDS if kw in normalized]
    simple_hits = [kw for kw in SIMPLE_KEYWORDS if kw in normalized]

    has_code_block = "```" in prompt
    has_json_like = "{" in prompt and "}" in prompt
    has_yaml_like = ":" in prompt and "\n" in prompt
    has_error_signal = any(x in normalized for x in ["error", "exception", "traceback", "failed", "fails"])
    has_question = "?" in prompt
    has_multi_step_signal = any(x in normalized for x in ["step", "plan", "first", "second", "third", "tradeoff", "option"])
    has_security_signal = any(x in normalized for x in ["security", "auth", "token", "jwt", "permission", "secret"])
    is_short_approval = normalized in APPROVAL_WORDS

    return {
        "word_count": word_count,
        "char_count": len(prompt),
        "complex_keyword_hits": complex_hits,
        "simple_keyword_hits": simple_hits,
        "has_code_block": has_code_block,
        "has_json_like": has_json_like,
        "has_yaml_like": has_yaml_like,
        "has_error_signal": has_error_signal,
        "has_question": has_question,
        "has_multi_step_signal": has_multi_step_signal,
        "has_security_signal": has_security_signal,
        "is_short_approval": is_short_approval,
    }


def rule_based_label(prompt: str) -> Tuple[str, float, str, Dict[str, Any]]:
    """
    Conservative pseudo-labeler.

    This is NOT human ground truth.
    It is only an initial labeling method so we can build the pipeline.
    When uncertain, it prefers escalate.
    """
    features = extract_features(prompt)

    score = 0
    reasons = []

    if features["is_short_approval"]:
        return (
            LABEL_CHEAP_OK,
            0.95,
            "Very short approval / continuation command.",
            features,
        )

    if features["word_count"] <= 5 and not features["has_error_signal"]:
        score -= 2
        reasons.append("Very short prompt.")

    if features["word_count"] >= 80:
        score += 2
        reasons.append("Long prompt.")

    if features["word_count"] >= 200:
        score += 2
        reasons.append("Very long prompt.")

    if features["complex_keyword_hits"]:
        score += min(4, len(features["complex_keyword_hits"]))
        reasons.append("Contains complex-domain keywords.")

    if features["simple_keyword_hits"] and features["word_count"] < 80:
        score -= 2
        reasons.append("Looks like a simple transformation or direct command.")

    if features["has_code_block"]:
        score += 1
        reasons.append("Contains code block.")

    if features["has_error_signal"]:
        score += 3
        reasons.append("Contains debugging/error signal.")

    if features["has_multi_step_signal"]:
        score += 2
        reasons.append("Contains planning or multi-step signal.")

    if features["has_security_signal"]:
        score += 3
        reasons.append("Contains security/auth signal.")

    if features["has_question"] and features["word_count"] > 40:
        score += 1
        reasons.append("Question with non-trivial context.")

    if score <= -1:
        label = LABEL_CHEAP_OK
        confidence = 0.75 if score == -1 else 0.88
    elif score >= 3:
        label = LABEL_ESCALATE
        confidence = min(0.95, 0.70 + (score * 0.04))
    else:
        label = LABEL_ESCALATE
        confidence = 0.62
        reasons.append("Borderline case; conservative default is escalate.")

    reason = " ".join(reasons) if reasons else "No strong signal; conservative default."

    return label, round(confidence, 4), reason, features


def label_prompt_dict(prompt_dict: Dict[str, Any]) -> Dict[str, Any]:
    prompt = prompt_dict.get("prompt", "") or ""
    label, confidence, reason, features = rule_based_label(prompt)

    updated = dict(prompt_dict)
    updated["difficulty_label"] = label
    updated["difficulty_confidence"] = confidence
    updated["labeling_reason"] = reason
    updated["labeling_method"] = "rule_based_pseudo_label_mvp_v1"
    updated["labeling_features"] = features
    return updated
