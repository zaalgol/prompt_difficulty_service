"""Custom Presidio anonymizer operator for coherent, realistic pseudonymization.

Adapted from Presidio's `InstanceCounterAnonymizer` pseudonymization sample
(https://microsoft.github.io/presidio/samples/python/pseudonymization). Instead of
emitting placeholder tags like `<PERSON_0>`, this operator generates a realistic
fake value with Faker the first time it sees an original value, then reuses that
stored value on every later occurrence.

Coherence is driven entirely by the shared `entity_mapping` dict passed in via the
operator params: because the same dict is reused for every prompt in a session, the
same original value always resolves to the same fake value — both within a single
prompt and across prompts.
"""
from typing import Callable, Dict

from faker import Faker
from presidio_anonymizer.operators import Operator, OperatorType

# A single Faker instance is reused. It is intentionally NOT seeded: a fixed seed
# would make every session produce identical fakes, and per-value consistency is
# already provided by the entity_mapping vault, not by the RNG.
_fake = Faker()

# Maps a Presidio entity type to a Faker generator that produces a plausible
# replacement of the same shape. Anything not listed falls back to a generic tag.
FAKER_BY_TYPE: Dict[str, Callable[[], str]] = {
    "PERSON": _fake.name,
    "EMAIL_ADDRESS": _fake.email,
    "PHONE_NUMBER": _fake.phone_number,
    "LOCATION": _fake.city,
    "NRP": _fake.country,
    "ORGANIZATION": _fake.company,
    "CREDIT_CARD": _fake.credit_card_number,
    "IBAN_CODE": _fake.iban,
    "US_SSN": _fake.ssn,
    "US_DRIVER_LICENSE": _fake.license_plate,
    "US_BANK_NUMBER": _fake.bban,
    "IP_ADDRESS": _fake.ipv4,
    "URL": _fake.url,
    "DATE_TIME": lambda: _fake.date(pattern="%Y-%m-%d"),
}


def _fake_value(entity_type: str) -> str:
    generator = FAKER_BY_TYPE.get(entity_type)
    if generator is None:
        return f"<{entity_type}>"
    return str(generator())


class ConsistentFakerAnonymizer(Operator):
    """Replace each PII value with a realistic, per-session-consistent fake value."""

    def operate(self, text: str, params: Dict = None) -> str:
        params = params or {}
        entity_type: str = params["entity_type"]
        # entity_mapping is a dict of dicts: {entity_type: {normalized_original: fake}}.
        # It is shared (and mutated) across every call for a given session, which
        # is what makes replacements consistent across prompts.
        entity_mapping: Dict[str, Dict[str, str]] = params["entity_mapping"]

        # Normalize the lookup key so case/whitespace variants of the same value
        # ("John Smith" vs "john  smith") resolve to one fake. Note: this cannot
        # fix differing NER span boundaries (e.g. "Email John Smith" vs
        # "John Smith"); that is a detection-level concern, not a mapping one.
        key = " ".join(text.split()).casefold()

        mapping_for_type = entity_mapping.setdefault(entity_type, {})
        if key in mapping_for_type:
            return mapping_for_type[key]

        new_value = self._unique_fake(entity_type, key, mapping_for_type)
        mapping_for_type[key] = new_value
        return new_value

    @staticmethod
    def _unique_fake(entity_type: str, key: str, mapping_for_type: Dict[str, str]) -> str:
        """Generate a fake distinct from the original and from prior fakes.

        Distinctness matters: if two different real values collapsed to one fake
        (Faker's pool is finite, so this happens), the anonymized text would lose
        the distinction between them. We also reject a fake equal to the original,
        which would leave the value un-anonymized.
        """
        used = set(mapping_for_type.values())
        for _ in range(10):
            candidate = _fake_value(entity_type)
            if candidate.casefold() != key and candidate not in used:
                return candidate
        # Exhausted retries (tiny/duplicated generator pool): disambiguate
        # deterministically so the result is still distinct.
        return f"{_fake_value(entity_type)} ({len(mapping_for_type) + 1})"

    def validate(self, params: Dict = None) -> None:
        params = params or {}
        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if not isinstance(params["entity_mapping"], dict):
            raise ValueError("`entity_mapping` must be a dict.")

    def operator_name(self) -> str:
        return "consistent_faker"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize
