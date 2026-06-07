"""A long, realistic multi-turn conversation exercised end-to-end through the
anonymizer, to prove the guarantees hold over a whole session rather than a
single call.

It interleaves: re-mentions of the same people and email (coherence), two
distinct people that must never collapse (distinctness), a date of birth reused
across several retirement/age questions (semantic preservation), and assorted
other PII (no leaks) — all under one session_id.

Uses the real Presidio stack; skipped when it (or its spaCy model) is missing.
Names are always preceded by a lowercase word so NER yields clean spans (the
boundary-drift caveat is covered separately in test_anonymize_behavior.py).
"""
from dataclasses import dataclass

import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")

SESSION = "long-conversation-1"
DOB = "1960-04-12"
JOHN_EMAIL = "john.smith@corp.com"


@pytest.fixture(scope="module")
def svc():
    from app.presidio_service import AnonymizerService

    s = AnonymizerService()
    try:
        s.anonymize("warmup for John Smith")
    except Exception as exc:  # pragma: no cover - environment without the model
        pytest.skip(f"Presidio engines unavailable: {exc}")
    return s


def _person_fake(result):
    span = next(e for e in result["entities"] if e["entity_type"] == "PERSON")
    return result["anonymized_prompt"][span["start"]:span["end"]]


def _email_fake(result):
    span = next(e for e in result["entities"] if e["entity_type"] == "EMAIL_ADDRESS")
    return result["anonymized_prompt"][span["start"]:span["end"]]


@dataclass
class Turn:
    prompt: str
    mentions_john: bool = False
    mentions_mary: bool = False
    mentions_email: bool = False
    preserve_date: bool = False  # a date-of-birth-dependent question


# A 16-turn conversation. The first three turns introduce each entity alone so
# its fake can be captured unambiguously; the rest weave them together.
CONVERSATION = [
    Turn("Please contact John Smith today.", mentions_john=True),
    Turn("Please contact Mary Johnson today.", mentions_mary=True),
    Turn(f"Please write to John Smith at {JOHN_EMAIL}.",
         mentions_john=True, mentions_email=True),
    Turn("Then loop in Mary Johnson on that thread.", mentions_mary=True),
    Turn(f"John Smith was born on {DOB}; when can he retire at 67?",
         mentions_john=True, preserve_date=True),
    Turn(f"Resend the invite to {JOHN_EMAIL} please.", mentions_email=True),
    Turn("Please ask Mary Johnson to review the draft.", mentions_mary=True),
    Turn(f"How old is John Smith if his birthday is {DOB}?",
         mentions_john=True, preserve_date=True),
    Turn("Please remind John Smith and Mary Johnson about Friday.",
         mentions_john=True, mentions_mary=True),
    Turn("Could you call Mary Johnson this afternoon?", mentions_mary=True),
    Turn(f"Confirm John Smith still uses {JOHN_EMAIL}.",
         mentions_john=True, mentions_email=True),
    Turn("Please tell John Smith the meeting moved.", mentions_john=True),
    Turn(f"Given the birth date {DOB}, when is John Smith eligible for a pension?",
         mentions_john=True, preserve_date=True),
    Turn("Finally, please thank Mary Johnson for her help.", mentions_mary=True),
    Turn("Please thank John Smith as well.", mentions_john=True),
    Turn(f"Summary: keep John Smith and Mary Johnson at {JOHN_EMAIL} in the loop.",
         mentions_john=True, mentions_mary=True, mentions_email=True),
]


@pytest.fixture(scope="module")
def conversation(svc):
    """Run the whole conversation once; return captured fakes + per-turn results."""
    john_fake = _person_fake(svc.anonymize(CONVERSATION[0].prompt, session_id=SESSION))
    mary_fake = _person_fake(svc.anonymize(CONVERSATION[1].prompt, session_id=SESSION))
    email_fake = _email_fake(svc.anonymize(CONVERSATION[2].prompt, session_id=SESSION))

    results = []
    for turn in CONVERSATION:
        results.append((turn, svc.anonymize(turn.prompt, session_id=SESSION)))

    return {
        "john_fake": john_fake,
        "mary_fake": mary_fake,
        "email_fake": email_fake,
        "results": results,
    }


# ── no leaks anywhere in the conversation ────────────────────────────────────

def test_no_real_pii_leaks_in_any_turn(conversation):
    for turn, result in conversation["results"]:
        out = result["anonymized_prompt"]
        assert "John Smith" not in out, f"LEAK in: {turn.prompt!r}"
        assert "Mary Johnson" not in out, f"LEAK in: {turn.prompt!r}"
        assert JOHN_EMAIL not in out, f"LEAK in: {turn.prompt!r}"


# ── coherence: every re-mention reuses the first fake ────────────────────────

def test_john_is_coherent_across_the_whole_conversation(conversation):
    john = conversation["john_fake"]
    for turn, result in conversation["results"]:
        if turn.mentions_john:
            assert john in result["anonymized_prompt"], \
                f"John's fake missing in: {turn.prompt!r}"


def test_mary_is_coherent_across_the_whole_conversation(conversation):
    mary = conversation["mary_fake"]
    for turn, result in conversation["results"]:
        if turn.mentions_mary:
            assert mary in result["anonymized_prompt"], \
                f"Mary's fake missing in: {turn.prompt!r}"


def test_email_is_coherent_across_the_whole_conversation(conversation):
    email = conversation["email_fake"]
    for turn, result in conversation["results"]:
        if turn.mentions_email:
            assert email in result["anonymized_prompt"], \
                f"Email fake missing in: {turn.prompt!r}"


# ── distinctness: the two people never collapse ──────────────────────────────

def test_two_people_have_distinct_fakes(conversation):
    assert conversation["john_fake"] != conversation["mary_fake"]


def test_turns_mentioning_both_keep_them_distinct(conversation):
    john, mary = conversation["john_fake"], conversation["mary_fake"]
    both_turns = [
        r for t, r in conversation["results"] if t.mentions_john and t.mentions_mary
    ]
    assert both_turns, "expected turns mentioning both people"
    for result in both_turns:
        out = result["anonymized_prompt"]
        assert john in out and mary in out


# ── semantics: the DOB is preserved exactly whenever the answer needs it ──────

def test_dob_preserved_in_every_date_dependent_turn(conversation):
    date_turns = [r for t, r in conversation["results"] if t.preserve_date]
    assert len(date_turns) == 3, "expected three date-dependent questions"
    for result in date_turns:
        assert "DATE_TIME" in result["preserved_entity_types"]
        assert DOB in result["anonymized_prompt"], "DOB was altered or dropped"
        dates = [e for e in result["entities"] if e["entity_type"] == "DATE_TIME"]
        assert dates and all(e["action"] == "preserved" for e in dates)


def test_name_still_anonymized_in_date_dependent_turns(conversation):
    # Preserving the date must not accidentally spare the person's name.
    for turn, result in conversation["results"]:
        if turn.preserve_date:
            assert conversation["john_fake"] in result["anonymized_prompt"]
            assert "John Smith" not in result["anonymized_prompt"]


# ── stability: replaying a turn yields identical output ──────────────────────

def test_replaying_a_turn_is_identical(svc, conversation):
    # The mapping built during the conversation makes re-anonymization deterministic.
    prompt = CONVERSATION[4].prompt  # the first retirement question
    again = svc.anonymize(prompt, session_id=SESSION)
    original = next(r for t, r in conversation["results"] if t.prompt == prompt)
    assert again["anonymized_prompt"] == original["anonymized_prompt"]


def test_a_fresh_session_does_not_inherit_the_mapping(svc, conversation):
    # Same prompt, new session: still anonymized, but not required to match the
    # other session's fake (mappings are per-session).
    out = svc.anonymize("Please contact John Smith today.",
                        session_id="long-conversation-2")["anonymized_prompt"]
    assert "John Smith" not in out
