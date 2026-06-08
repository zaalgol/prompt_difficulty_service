"""Deep behavioral tests for anonymization against the real Presidio stack.

Organized around the two failure directions the feature must avoid:
  - LEAK: PII that should be anonymized survives in the output.
  - SEMANTIC LOSS: a value the answer depends on gets changed, or coherence /
    distinctness across a conversation breaks.

A single module-scoped AnonymizerService loads the spaCy model once. Tests call
the service directly (no HTTP) so they stay focused on the anonymization logic.
"""
import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")


@pytest.fixture(scope="module")
def svc():
    from app.presidio_service import AnonymizerService

    s = AnonymizerService()
    try:
        s.anonymize("warmup for John Smith")  # triggers the lazy engine load
    except Exception as exc:  # pragma: no cover - environment without the model
        pytest.skip(f"Presidio engines unavailable: {exc}")
    return s


def person_fake(result):
    """The substring the first PERSON entity was replaced with."""
    span = next(e for e in result["entities"] if e["entity_type"] == "PERSON")
    return result["anonymized_prompt"][span["start"]:span["end"]]


# ── LEAK direction: detection + replacement coverage ─────────────────────────

@pytest.mark.parametrize("text, needle, etype", [
    ("Reach me at alice.jones@example.com anytime.", "alice.jones@example.com", "EMAIL_ADDRESS"),
    ("My Visa is 4111 1111 1111 1111 thanks.", "4111 1111 1111 1111", "CREDIT_CARD"),
    ("The server lives at 192.168.10.45 internally.", "192.168.10.45", "IP_ADDRESS"),
    # SSN detection is number-dependent in Presidio; 457-55-5462 scores reliably
    # (078-05-1120 does not — see the documented-gap section below).
    ("My SSN is 457-55-5462 for the form.", "457-55-5462", "US_SSN"),
    ("Docs are at https://secret.example.com/report for review.", "https://secret.example.com/report", "URL"),
])
def test_pii_is_detected_and_removed(svc, text, needle, etype):
    result = svc.anonymize(text)
    assert needle not in result["anonymized_prompt"], f"LEAK: {needle!r} survived"
    assert any(e["entity_type"] == etype for e in result["entities"]), \
        f"{etype} not detected in {text!r}"


def test_phone_number_is_removed(svc):
    result = svc.anonymize("Call me at (212) 555-0182 after noon.")
    assert "555-0182" not in result["anonymized_prompt"]


def test_multiple_pii_types_all_removed(svc):
    text = ("I'm John Smith, email john@corp.com, card 4111 1111 1111 1111, "
            "IP 10.0.0.5.")
    result = svc.anonymize(text)
    out = result["anonymized_prompt"]
    for needle in ["John Smith", "john@corp.com", "4111 1111 1111 1111", "10.0.0.5"]:
        assert needle not in out, f"LEAK: {needle!r} survived"


def test_pii_inside_json_is_removed(svc):
    text = '{"name": "John Smith", "email": "john@corp.com"}'
    out = svc.anonymize(text)["anonymized_prompt"]
    assert "John Smith" not in out
    assert "john@corp.com" not in out


def test_unicode_name_handling(svc):
    # If detected, it must be replaced; assert no leak rather than forcing NER.
    result = svc.anonymize("My colleague José García will join the call.")
    if any(e["entity_type"] == "PERSON" for e in result["entities"]):
        assert "José García" not in result["anonymized_prompt"]


def test_prompt_with_no_pii_is_unchanged(svc):
    # Avoids capitalized acronyms (YAML/JSON), which Presidio NER tends to
    # mis-tag as ORGANIZATION — see the documented-gap section below.
    text = "Please reformat the configuration and fix the indentation."
    result = svc.anonymize(text)
    assert result["anonymized_prompt"] == text
    assert result["entities"] == []


# ── SEMANTIC-LOSS direction: coherence ───────────────────────────────────────

def test_coherence_within_prompt(svc):
    result = svc.anonymize("John Smith met John Smith and John Smith agreed.")
    out = result["anonymized_prompt"]
    assert "John Smith" not in out
    persons = [e for e in result["entities"] if e["entity_type"] == "PERSON"]
    assert len(persons) == 3
    # All three replaced spans must read as the same fake name.
    fakes = {out[e["start"]:e["end"]] for e in persons}
    assert len(fakes) == 1, f"coherence broke within a prompt: {fakes}"


def test_coherence_across_many_turns(svc):
    # Each turn keeps "John Smith" preceded by a lowercase word so NER yields a
    # clean "John Smith" span (capitalized leading tokens can get absorbed into
    # the span — that boundary-drift case is documented separately below).
    sid = "behavior-multiturn"
    r1 = svc.anonymize("Please email John Smith.", session_id=sid)
    fake = person_fake(r1)
    for follow_up in [
        "Did John Smith reply?",
        "Please ask John Smith again.",
        "We should thank John Smith.",
    ]:
        out = svc.anonymize(follow_up, session_id=sid)["anonymized_prompt"]
        assert fake in out, "coherence broke across turns"
        assert "John Smith" not in out


def test_coherence_for_email_across_turns(svc):
    sid = "behavior-email"
    r1 = svc.anonymize("Write to bob@corp.com now.", session_id=sid)
    span = next(e for e in r1["entities"] if e["entity_type"] == "EMAIL_ADDRESS")
    fake_email = r1["anonymized_prompt"][span["start"]:span["end"]]
    out2 = svc.anonymize("Resend to bob@corp.com please.", session_id=sid)["anonymized_prompt"]
    assert fake_email in out2
    assert "bob@corp.com" not in out2


def test_case_variant_is_coherent_across_turns(svc):
    # Identical surrounding context differing only in case, so the only variable
    # under test is the operator's casefold normalization (not span boundaries).
    sid = "behavior-case"
    r1 = svc.anonymize("Please page John Smith now.", session_id=sid)
    fake = person_fake(r1)
    out2 = svc.anonymize("Please page JOHN SMITH now.", session_id=sid)["anonymized_prompt"]
    assert fake in out2


def test_distinct_people_get_distinct_fakes(svc):
    result = svc.anonymize("John Smith emailed Mary Johnson about Peter Brown.")
    persons = [e for e in result["entities"] if e["entity_type"] == "PERSON"]
    assert len(persons) == 3
    out = result["anonymized_prompt"]
    fakes = {out[e["start"]:e["end"]] for e in persons}
    assert len(fakes) == 3, f"distinct people collapsed to {fakes}"


def test_different_sessions_isolated(svc):
    r1 = svc.anonymize("Email John Smith.", session_id="iso-a")
    r2 = svc.anonymize("Email John Smith.", session_id="iso-b")
    assert "John Smith" not in r1["anonymized_prompt"]
    assert "John Smith" not in r2["anonymized_prompt"]


def test_idempotent_within_session(svc):
    sid = "behavior-idem"
    text = "John Smith at john@corp.com, IP 10.0.0.5."
    first = svc.anonymize(text, session_id=sid)
    second = svc.anonymize(text, session_id=sid)
    assert first["anonymized_prompt"] == second["anonymized_prompt"]
    assert first["entities"] == second["entities"]


# ── SEMANTIC-LOSS direction: auto-preserve correctness ───────────────────────

PRESERVE_PROMPTS = [
    "I am John Smith, born 1960-04-12. When do I retire at 67?",
    "John Smith here — how old am I if born on 1985-07-09?",
    "How many days until my visa, issued 2020-01-01, expires on 2025-01-01?",
]


@pytest.mark.parametrize("prompt", PRESERVE_PROMPTS)
def test_auto_preserve_keeps_every_date_in_intent_prompts(svc, prompt):
    result = svc.anonymize(prompt)
    assert "DATE_TIME" in result["preserved_entity_types"]
    # Every detected DATE_TIME span must be marked preserved...
    dates = [e for e in result["entities"] if e["entity_type"] == "DATE_TIME"]
    assert dates, f"no date detected in {prompt!r}"
    assert all(e["action"] == "preserved" for e in dates)


def test_preserved_date_is_byte_identical(svc):
    prompt = "Born 1960-04-12, when do I retire?"
    out = svc.anonymize(prompt)["anonymized_prompt"]
    assert "1960-04-12" in out  # not reformatted, not shifted


def test_name_still_anonymized_when_date_preserved(svc):
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    out = svc.anonymize(prompt)["anonymized_prompt"]
    assert "1960-04-12" in out
    assert "Jane Doe" not in out  # date kept, but the person is not


def test_multiple_dates_all_preserved(svc):
    prompt = ("I, John Smith, was born 1960-04-12 and was hired 1985-06-01. "
              "When do I retire?")
    result = svc.anonymize(prompt)
    out = result["anonymized_prompt"]
    assert "1960-04-12" in out
    assert "1985-06-01" in out
    assert "John Smith" not in out


def test_date_without_intent_is_anonymized(svc):
    prompt = "My name is Jane Doe and I last logged in on 2021-03-08."
    result = svc.anonymize(prompt)
    assert "DATE_TIME" not in result["preserved_entity_types"]
    assert "2021-03-08" not in result["anonymized_prompt"]  # treated as PII
    assert "Jane Doe" not in result["anonymized_prompt"]


def test_auto_preserve_disabled_anonymizes_the_date(svc):
    prompt = "I am Jane Doe, born 1960-04-12. When do I retire at 67?"
    result = svc.anonymize(prompt, auto_preserve=False)
    assert result["preserved_entity_types"] == []
    assert "1960-04-12" not in result["anonymized_prompt"]


def test_explicit_preserve_still_works(svc):
    prompt = "Jane Doe logged in on 2021-03-08."  # no intent keyword
    result = svc.anonymize(prompt, preserve_entity_types=["DATE_TIME"])
    assert "DATE_TIME" in result["preserved_entity_types"]
    assert "2021-03-08" in result["anonymized_prompt"]


def test_explicit_and_auto_preserve_union(svc):
    # EMAIL forced explicitly; DATE_TIME inferred from the retirement question.
    prompt = "I am Jane Doe (jane@corp.com), born 1960-04-12, when do I retire?"
    result = svc.anonymize(prompt, preserve_entity_types=["EMAIL_ADDRESS"])
    assert set(["EMAIL_ADDRESS", "DATE_TIME"]).issubset(set(result["preserved_entity_types"]))
    out = result["anonymized_prompt"]
    assert "jane@corp.com" in out   # explicitly preserved
    assert "1960-04-12" in out      # auto preserved
    assert "Jane Doe" not in out    # still anonymized


def test_preserving_dates_does_not_block_other_pii(svc):
    prompt = ("I am John Smith, email john@corp.com, born 1960-04-12, "
              "when do I retire?")
    out = svc.anonymize(prompt)["anonymized_prompt"]
    assert "1960-04-12" in out        # preserved
    assert "John Smith" not in out    # anonymized
    assert "john@corp.com" not in out # anonymized


# ── edge cases ───────────────────────────────────────────────────────────────

def test_long_prompt_keeps_coherence(svc):
    sid = "behavior-long"
    filler = "This is some neutral filler text about scheduling and logistics. " * 40
    prompt = f"{filler} Please contact John Smith. {filler} And cc John Smith."
    result = svc.anonymize(prompt, session_id=sid)
    out = result["anonymized_prompt"]
    persons = [e for e in result["entities"] if e["entity_type"] == "PERSON"]
    fakes = {out[e["start"]:e["end"]] for e in persons}
    assert "John Smith" not in out
    assert len(fakes) == 1


def test_repeated_identical_prompts_are_stable(svc):
    sid = "behavior-stable"
    text = "Contact John Smith at john@corp.com."
    outputs = {svc.anonymize(text, session_id=sid)["anonymized_prompt"] for _ in range(5)}
    assert len(outputs) == 1  # deterministic within the session


def test_entities_sorted_left_to_right(svc):
    result = svc.anonymize("John Smith emailed Mary Johnson and Peter Brown.")
    starts = [e["start"] for e in result["entities"]]
    assert starts == sorted(starts)


# ── Documented Presidio limitations (current behavior, locked in) ─────────────
# These pin known weak spots by asserting what the system does *today*, so the
# gaps are visible and any future change is caught. Each marks a way the system
# can silently fail in one of the two dangerous directions. If detection is later
# hardened (custom recognizers, lower thresholds, post-checks), the matching test
# here will start failing — that is the signal to update it and close the gap.

def test_gap_some_ssns_are_not_detected(svc):
    # KNOWN LIMITATION (LEAK): 078-05-1120 (the famous invalid promo SSN) is not
    # flagged at threshold 0.5, so the digits survive — i.e. they are NOT
    # anonymized. Asserting the leak documents the gap; fix = custom SSN recognizer.
    out = svc.anonymize("My SSN is 078-05-1120 for the form.")["anonymized_prompt"]
    assert "078-05-1120" in out, "SSN is now detected — promote this to a no-leak assertion"


def test_gap_boundary_drift_breaks_coherence(svc):
    # KNOWN LIMITATION (COHERENCE): "Email John Smith" is captured as one PERSON
    # span, so its vault key differs from a clean "John Smith" and the two turns
    # get *different* fakes. Asserting the divergence documents the gap.
    sid = "gap-boundary"
    fake_absorbed = svc.anonymize("Email John Smith.", session_id=sid)["anonymized_prompt"].rstrip(".")
    out_clean = svc.anonymize("Please email John Smith.", session_id=sid)["anonymized_prompt"]
    assert fake_absorbed not in out_clean, "boundary drift resolved — coherence now holds"


def test_technical_acronyms_are_not_over_detected(svc):
    prompt = "Please reformat this YAML file."
    result = svc.anonymize(prompt)
    assert result["anonymized_prompt"] == prompt
    assert not any(e["entity_type"] == "ORGANIZATION" for e in result["entities"])
