"""Additional decision-boundary tests for semantic auto-preservation."""
import pytest

from app.presidio_service.semantics import infer_required_entity_types


@pytest.mark.parametrize(
    "prompt",
    [
        "RETIREMENT date from 1960-04-12?",
        "How\nmany\ndays until 2027-01-01?",
        "At what age should I claim my pension?",
        "My passport expires 2028-09-12; WHEN SHOULD I RENEW IT?",
        "There are 14 days remaining until 2026-08-20.",
        "Calculate the countdown to 2026-12-31.",
        "What is the due date?",
        "How long has it been since 2014-05-06?",
        "How many months till the anniversary?",
        "When would their membership expire?",
        "When are you eligible based on your age?",
    ],
)
def test_date_dependency_variants_preserve_date_time(prompt):
    assert infer_required_entity_types(prompt) == ["DATE_TIME"]


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain renewable energy policy from 2020.",
        "Rename the renewables module created on 2021-01-01.",
        "The oldest supported API was released on 2019-10-10.",
        "Use the ageless color palette from the 1990s.",
        "The deadlineExceeded variable was added on 2022-02-02.",
        "Summarize expiration policy document dated 2020-03-03.",
        "The countdown_widget test failed on 2024-04-04.",
        "Discuss pension fund architecture from 2023.",
    ],
)
def test_near_miss_words_do_not_preserve_dates(prompt):
    assert infer_required_entity_types(prompt) == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Do not calculate how old I am; anonymize the DOB 1960-04-12.",
        "The quoted phrase 'when do I retire' is test data from 2020-01-01.",
        "Check whether the text says retirement, but redact every date.",
        "Explain why 'days until' is a phrase; the sample date is 2025-05-05.",
    ],
)
def test_gap_negated_or_quoted_intent_still_preserves(prompt):
    """Document the current keyword heuristic's lack of negation/quotation."""
    assert infer_required_entity_types(prompt) == ["DATE_TIME"], (
        "Negated/meta intent is now understood; promote to a no-preserve assertion"
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Born 1980-01-01. Subtract that year from the current year.",
        "Compare 2026-06-07 with my birth date 1990-03-02.",
        "Compute current_year - 1975 for me.",
        "DOB: 1960-04-12. Add 67 years and return the resulting date.",
    ],
)
def test_gap_implicit_date_arithmetic_is_not_preserved(prompt):
    """Document implicit arithmetic that the current heuristic misses."""
    assert infer_required_entity_types(prompt) == [], (
        "Implicit date arithmetic is now recognized; promote to a preserve assertion"
    )
