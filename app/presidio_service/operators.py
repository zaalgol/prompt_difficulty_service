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
import threading
import unicodedata
from typing import Callable, Dict

from faker import Faker
from presidio_anonymizer.operators import Operator, OperatorType

from app.config import PRESIDIO_MAX_ENTRIES_PER_TYPE

# A single Faker instance is reused. It is intentionally NOT seeded: a fixed seed
# would make every session produce identical fakes, and per-value consistency is
# already provided by the entity_mapping vault, not by the RNG.
_fake = Faker()
_mapping_lock = threading.RLock()


# Replacements for network-/system-addressable types use reserved, non-routable
# values so the anonymized prompt can never point an LLM (or an agent that acts on
# its output) at a real host, mailbox, or phone line. See RFC 2606 (example.com /
# .invalid), RFC 5737 (192.0.2.0/24 documentation IPs), and the NANP 555-01xx
# fictional-number range.
def _safe_email() -> str:
    return f"{_fake.user_name()}@example.com"


def _safe_url() -> str:
    return f"https://example.com/{_fake.uri_path()}"


def _safe_ip() -> str:
    # 192.0.2.0/24 is reserved for documentation and is not routable.
    return f"192.0.2.{_fake.random_int(min=1, max=254)}"


def _safe_phone() -> str:
    # 555-0100..555-0199 are reserved for fictional use.
    return f"(555) 555-01{_fake.random_int(min=0, max=99):02d}"


# Maps a Presidio entity type to a generator that produces a plausible, *safe*
# replacement of the same shape. Anything not listed falls back to a generic tag.
FAKER_BY_TYPE: Dict[str, Callable[[], str]] = {
    "PERSON": _fake.name,
    "EMAIL_ADDRESS": _safe_email,
    "PHONE_NUMBER": _safe_phone,
    "LOCATION": _fake.city,
    "NRP": _fake.country,
    "ORGANIZATION": _fake.company,
    "CREDIT_CARD": _fake.credit_card_number,
    "IBAN_CODE": _fake.iban,
    "US_SSN": _fake.ssn,
    "US_DRIVER_LICENSE": _fake.license_plate,
    "US_BANK_NUMBER": _fake.bban,
    "IP_ADDRESS": _safe_ip,
    "URL": _safe_url,
    "DATE_TIME": lambda: _fake.date(pattern="%Y-%m-%d"),
}


def _fake_value(entity_type: str) -> str:
    generator = FAKER_BY_TYPE.get(entity_type)
    if generator is None:
        return f"<{entity_type}>"
    return str(generator())


def _mapping_key(value: str) -> str:
    """Normalize equivalent user spellings to one stable vault key."""
    collapsed = " ".join(value.split())
    return unicodedata.normalize("NFKC", collapsed).casefold()


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
        key = _mapping_key(text)

        # The mapping is shared by concurrent FastAPI worker threads. Keep the
        # check/generate/store sequence atomic so two first sightings cannot
        # assign different pseudonyms to the same original.
        with _mapping_lock:
            mapping_for_type = entity_mapping.setdefault(entity_type, {})
            if key in mapping_for_type:
                return mapping_for_type[key]

            new_value = self._unique_fake(entity_type, key, mapping_for_type)
            # Bound per-session memory: drop the oldest mapping for this type once
            # the cap is reached (dict preserves insertion order). A caller pumping
            # unique values into one session can no longer grow it without limit;
            # the trade-off is that very old values may lose coherence.
            if len(mapping_for_type) >= PRESIDIO_MAX_ENTRIES_PER_TYPE:
                mapping_for_type.pop(next(iter(mapping_for_type)))
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
        used = {_mapping_key(value) for value in mapping_for_type.values()}
        for _ in range(10):
            candidate = _fake_value(entity_type)
            candidate_key = _mapping_key(candidate)
            if candidate_key != key and candidate_key not in used:
                return candidate
        # Never suffix a rejected candidate: it may contain the original value.
        # Use an opaque deterministic fallback and probe until it is unique.
        suffix = len(mapping_for_type) + 1
        while True:
            candidate = f"<{entity_type}_{suffix}>"
            candidate_key = _mapping_key(candidate)
            if candidate_key != key and candidate_key not in used:
                return candidate
            suffix += 1

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
