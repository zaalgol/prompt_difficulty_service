"""Unit tests for the auto-preservation intent heuristic (no Presidio needed).

This is the highest-stakes piece of the feature: a false positive leaks a real
date that did not need keeping, a false negative anonymizes a date the answer
depended on. The tables below pin the decision boundary in both directions.
"""
import pytest

from app.presidio_service.semantics import infer_required_entity_types

# Prompts whose answer is computed from a date/age in the text -> keep DATE_TIME.
SHOULD_PRESERVE_DATES = [
    "I was born on 1960-04-12, when do I retire at 67?",
    "When will I retire if I started working in 1985?",
    "My pension starts soon — what date exactly?",
    "How old am I if I was born in 1990?",
    "What is my age?",
    "Tell me his age based on his birth year 1975.",
    "How many days until 2026-12-25?",
    "How many years until my mortgage is paid off?",
    "How long until my visa expires?",
    "When does my passport expire?",
    "My membership expires when?",
    "How long until my renewal date of 2025-09-01?",
    "Please renew it — when is the deadline?",
    "What is the due date for the 2025-01-10 invoice?",
    "When is my wedding anniversary if we married on 2010-06-06?",
    "Start a countdown to my birthday 1995-03-03.",
    "When should I renew my license?",
    "How many weeks until the launch on 2025-05-01?",
    "Given my birthday, when am I eligible for Medicare?",
    "From my date of birth, when do I turn 65?",  # "when do" intent
]

# Prompts that merely contain a date but do not compute from it -> anonymize it.
SHOULD_NOT_PRESERVE = [
    "Summarize this article about John Smith.",
    "Translate 'hello' to French.",
    "My name is Jane Doe and I last logged in on 2021-03-08.",
    "Draft an email to the team about the 2021-03-08 outage.",
    "Reschedule the meeting that was on 2020-01-01.",
    "Tell me about renewable energy trends in 2020.",  # 'renewable' must NOT match
    "Fix the bug in the function I wrote on 2019-12-31.",
    "Here is a log entry from 2022-07-04, classify its severity.",
    "Write a poem mentioning the year 1999.",
    "What happened on 1969-07-20 in history?",  # factual, not a calc on user PII
]


@pytest.mark.parametrize("prompt", SHOULD_PRESERVE_DATES)
def test_date_intent_preserves_date_time(prompt):
    assert "DATE_TIME" in infer_required_entity_types(prompt), prompt


@pytest.mark.parametrize("prompt", SHOULD_NOT_PRESERVE)
def test_no_date_intent_preserves_nothing(prompt):
    assert infer_required_entity_types(prompt) == [], prompt


def test_returns_plain_list_of_strings():
    out = infer_required_entity_types("when do I retire?")
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)


def test_empty_and_whitespace_prompts():
    assert infer_required_entity_types("") == []
    assert infer_required_entity_types("   \n\t ") == []


def test_case_insensitive():
    assert "DATE_TIME" in infer_required_entity_types("WHEN DO I RETIRE?")
    assert "DATE_TIME" in infer_required_entity_types("How Old Am I?")


# ── Known limitations: documented blind spots of a keyword heuristic ──────────
# These pin the heuristic's current behavior so the 80%->99.9% distance is
# visible instead of silently shipped. They assert what happens today; if the
# heuristic is later improved, the matching test fails — the signal to update it.

def test_gap_implicit_age_calc_without_keyword_is_missed():
    # KNOWN LIMITATION: the answer depends on the birth year, but no trigger word
    # appears, so nothing is preserved and the date would be anonymized.
    assert infer_required_entity_types(
        "I was born in 1980. Compute the gap between then and today."
    ) == [], "heuristic now catches implicit intent — promote to a preserve assertion"


def test_gap_retire_without_any_date_is_a_false_positive():
    # KNOWN LIMITATION: the trigger is lexical, so 'retire' preserves DATE_TIME
    # even when no date is present (harmless but imprecise).
    assert infer_required_entity_types("Should I retire early or keep working?") == ["DATE_TIME"]
