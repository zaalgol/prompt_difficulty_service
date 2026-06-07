"""Complex invariants against the real Presidio analyzer/anonymizer stack."""
import json
import re

import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")


@pytest.fixture(scope="module")
def svc():
    from app.presidio_service import AnonymizerService

    service = AnonymizerService()
    try:
        service.anonymize("warmup for John Smith")
    except Exception as exc:  # pragma: no cover - environment without NLP model
        pytest.skip(f"Presidio engines unavailable: {exc}")
    return service


def _entity_texts(result, entity_type):
    output = result["anonymized_prompt"]
    return [
        output[item["start"]:item["end"]]
        for item in result["entities"]
        if item["entity_type"] == entity_type
    ]


@pytest.mark.parametrize(
    "prompt, originals, required_types",
    [
        (
            "Email first.last+alerts@example.co.uk or backup_user@sub.example.com.",
            ["first.last+alerts@example.co.uk", "backup_user@sub.example.com"],
            {"EMAIL_ADDRESS"},
        ),
        (
            "Traffic came from 203.0.113.42 and 2001:db8:85a3::8a2e:370:7334.",
            ["203.0.113.42", "2001:db8:85a3::8a2e:370:7334"],
            {"IP_ADDRESS"},
        ),
        (
            "Use cards 4111-1111-1111-1111 and 5555 5555 5555 4444.",
            ["4111-1111-1111-1111", "5555 5555 5555 4444"],
            {"CREDIT_CARD"},
        ),
        (
            "References: https://private.example.com/a?user=7 and http://10.0.0.8/admin.",
            ["https://private.example.com/a?user=7", "10.0.0.8"],
            {"URL", "IP_ADDRESS"},
        ),
    ],
)
def test_dense_pattern_pii_is_removed(svc, prompt, originals, required_types):
    result = svc.anonymize(prompt)
    output = result["anonymized_prompt"]
    detected = {item["entity_type"] for item in result["entities"]}

    for original in originals:
        assert original not in output, f"LEAK: {original!r} survived in {output!r}"
    assert required_types <= detected


def test_gap_international_phone_format_is_not_detected(svc):
    """Document the current UK-number leak until a recognizer closes it."""
    prompt = "Call +1 (212) 555-0182 or +44 20 7946 0958."

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "+44 20 7946 0958" in output, (
        "International phone is now detected; promote this to a no-leak assertion"
    )


def test_json_remains_parseable_and_non_pii_fields_remain_equal(svc):
    original = {
        "event": "password_reset",
        "actor": {
            "email": "john.smith+prod@example.com",
            "ip": "203.0.113.42",
        },
        "success": False,
        "attempts": 3,
        "tags": ["security", "interactive"],
    }
    prompt = json.dumps(original, separators=(",", ":"))

    result = svc.anonymize(prompt, session_id="structured-json")
    parsed = json.loads(result["anonymized_prompt"])

    assert parsed["event"] == original["event"]
    assert parsed["success"] is False
    assert parsed["attempts"] == 3
    assert parsed["tags"] == original["tags"]
    assert parsed["actor"]["email"] != original["actor"]["email"]
    assert parsed["actor"]["ip"] != original["actor"]["ip"]


def test_gap_person_name_inside_compact_json_is_not_detected(svc):
    """Document that compact JSON context can currently hide a person name."""
    prompt = '{"name":"John Smith","role":"approver"}'

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "John Smith" in output, (
        "Compact JSON person is now detected; promote this to a no-leak assertion"
    )
    assert json.loads(output)["role"] == "approver"


def test_markdown_field_boundaries_survive_person_replacement(svc):
    prompt = "- Owner: John Smith\n- Team: platform\n"

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "- Owner:" in output
    assert "- Team: platform" in output
    assert "John Smith" not in output


def test_gap_person_span_absorbs_following_pii_field_label(svc):
    """Document the semantic damage caused by this current NER boundary."""
    prompt = "- Owner: John Smith\n- Email: john@example.com\n"

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "- Owner:" in output
    assert "- Email:" not in output, (
        "Markdown field boundary is now stable; promote to a preservation assertion"
    )
    assert "John Smith" not in output
    assert "john@example.com" not in output


def test_markdown_and_code_structure_survives_replacement(svc):
    prompt = (
        "# Incident\n\n"
        "- Status: open\n"
        "- Email: john@example.com\n\n"
        "```json\n"
        '{"ip":"203.0.113.42","retry":3}\n'
        "```\n"
    )

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert output.startswith("# Incident\n\n")
    assert "- Status: open" in output
    assert "- Email:" in output
    assert "```json\n" in output
    assert '"retry":3' in output
    assert output.endswith("```\n")
    assert "john@example.com" not in output
    assert "203.0.113.42" not in output


def test_entity_offsets_are_sorted_non_overlapping_and_in_bounds(svc):
    result = svc.anonymize(
        "John Smith emailed jane.doe@example.com from 203.0.113.42 "
        "and used card 4111 1111 1111 1111."
    )
    output = result["anonymized_prompt"]
    entities = result["entities"]

    assert entities
    assert entities == sorted(entities, key=lambda item: item["start"])
    previous_end = 0
    for item in entities:
        assert 0 <= item["start"] < item["end"] <= len(output)
        assert item["start"] >= previous_end
        assert output[item["start"]:item["end"]]
        assert item["action"] in {"anonymized", "preserved"}
        previous_end = item["end"]


def test_repeated_mixed_entities_preserve_equality_relationships(svc):
    prompt = (
        "John Smith uses john@example.com. John Smith confirmed that "
        "john@example.com is still correct, while Mary Johnson uses "
        "mary@example.com."
    )
    result = svc.anonymize(prompt, session_id="mixed-equality")

    people = _entity_texts(result, "PERSON")
    emails = _entity_texts(result, "EMAIL_ADDRESS")
    assert len(people) == 3
    assert people[0] == people[1]
    assert people[0] != people[2]
    assert len(emails) == 3
    assert emails[0] == emails[1]
    assert emails[0] != emails[2]


def test_preserved_values_are_exact_while_adjacent_pii_changes(svc):
    prompt = (
        "Jane Doe (jane@example.com) was born on 1960-04-12; "
        "when will she turn 67?"
    )

    result = svc.anonymize(prompt)
    output = result["anonymized_prompt"]

    assert "1960-04-12" in output
    assert "Jane Doe" not in output
    assert "jane@example.com" not in output
    dates = [
        item for item in result["entities"] if item["entity_type"] == "DATE_TIME"
    ]
    assert dates and all(item["action"] == "preserved" for item in dates)


def test_gap_date_preservation_is_type_wide(svc):
    """Document that preserving one required date currently preserves all dates."""
    prompt = (
        "My DOB is 1960-04-12 and my last private login was 2024-11-03. "
        "When do I retire at 67?"
    )

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "1960-04-12" in output
    assert "2024-11-03" in output, (
        "Unrelated date is now anonymized; promote this to a selective-preservation test"
    )


def test_gap_person_mapping_changes_across_sentence_positions(svc):
    """Document current coherence loss caused by differing NER boundaries."""
    session = "boundary-position"
    first = svc.anonymize("Email John Smith.", session_id=session)
    second = svc.anonymize("Please email John Smith.", session_id=session)

    assert _entity_texts(first, "PERSON") != _entity_texts(second, "PERSON"), (
        "Sentence-position coherence is fixed; promote to an equality assertion"
    )


def test_output_does_not_contain_common_placeholder_failure_artifacts(svc):
    result = svc.anonymize(
        "John Smith at john@example.com called from 203.0.113.42."
    )

    assert not re.search(r"<(?:PERSON|EMAIL_ADDRESS|IP_ADDRESS)>", result["anonymized_prompt"])
