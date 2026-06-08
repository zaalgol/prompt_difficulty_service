"""Robustness tests for credentials and developer-sensitive data.

All values in this file are synthetic test fixtures and are not live credentials.
The tests are strict: a matching secret must not survive in anonymized output.
"""
import json
import re

import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")
pytest.importorskip("faker")


OPENAI_KEY = "sk-proj-FAKEabcdefghijklmnopqrstuvwxyz0123456789"
ANTHROPIC_KEY = "sk-ant-api03-FAKEabcdefghijklmnopqrstuvwxyz012345"
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
GITHUB_TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
GITHUB_FINE_GRAINED = "github_pat_FAKE_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
GITLAB_TOKEN = "glpat-FAKEabcdefghijklmnopqrstuv"
SLACK_TOKEN = "xoxb-123456789012-123456789012-FAKEabcdefgh"
STRIPE_KEY = "sk_test_FAKEabcdefghijklmnopqrstuv"
GOOGLE_KEY = "AIzaSyD_FAKEabcdefghijklmnopqrstuvwxyz123456"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIiwicm9sZSI6ImFkbWluIn0."
    "FAKEsignature0123456789"
)
DATABASE_PASSWORD = "FAKE-db-Passw0rd!"
GENERIC_PASSWORD = "CorrectHorseBatteryStaple_FAKE"
SESSION_COOKIE = "FAKEsession0123456789abcdef"
AZURE_ACCOUNT_KEY = "RkFLRUFDQ09VTlRLRVlGT1JURVNUU09OTFk="
PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\n"
    "RkFLRS1QUklWQVRFLUtFWS1EQVRBLU5PVC1SRUFM\n"
    "-----END PRIVATE KEY-----"
)


@pytest.fixture(scope="module")
def svc():
    from app.presidio_service import AnonymizerService

    service = AnonymizerService()
    try:
        service.warmup()
    except Exception as exc:  # pragma: no cover - environment without NLP model
        pytest.skip(f"Presidio engines unavailable: {exc}")
    return service


@pytest.mark.parametrize(
    "label, prompt, secret",
    [
        ("OpenAI", f"OPENAI_API_KEY={OPENAI_KEY}", OPENAI_KEY),
        ("Anthropic", f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}", ANTHROPIC_KEY),
        ("AWS access", f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY}", AWS_ACCESS_KEY),
        ("AWS secret", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_KEY}", AWS_SECRET_KEY),
        ("GitHub classic", f"GITHUB_TOKEN={GITHUB_TOKEN}", GITHUB_TOKEN),
        ("GitHub fine-grained", f"token: {GITHUB_FINE_GRAINED}", GITHUB_FINE_GRAINED),
        ("GitLab", f"PRIVATE_TOKEN={GITLAB_TOKEN}", GITLAB_TOKEN),
        ("Slack", f"SLACK_BOT_TOKEN={SLACK_TOKEN}", SLACK_TOKEN),
        ("Stripe", f"STRIPE_SECRET_KEY={STRIPE_KEY}", STRIPE_KEY),
        ("Google", f"GOOGLE_API_KEY={GOOGLE_KEY}", GOOGLE_KEY),
        ("JWT", f"jwt={JWT}", JWT),
        ("Bearer", f"Authorization: Bearer {GITHUB_TOKEN}", GITHUB_TOKEN),
        ("Basic auth", "Authorization: Basic ZmFrZS11c2VyOmZha2UtcGFzc3dvcmQ=", "ZmFrZS11c2VyOmZha2UtcGFzc3dvcmQ="),
        ("Password", f'password = "{GENERIC_PASSWORD}"', GENERIC_PASSWORD),
        ("Azure account", f"AccountKey={AZURE_ACCOUNT_KEY}", AZURE_ACCOUNT_KEY),
        ("Cookie", f"Cookie: sessionid={SESSION_COOKIE}; Path=/", SESSION_COOKIE),
        ("Private key", f"Deploy with:\n{PRIVATE_KEY}\nDo not log it.", PRIVATE_KEY),
    ],
)
def test_provider_and_generic_secrets_are_removed(svc, label, prompt, secret):
    result = svc.anonymize(prompt, session_id=f"secret-{label}")

    assert secret not in result["anonymized_prompt"], (
        f"{label} secret leaked: {result['anonymized_prompt']!r}"
    )
    assert result["entities"], f"{label} secret was not detected"


@pytest.mark.parametrize(
    "scheme",
    ["postgresql", "mysql", "mongodb", "mongodb+srv", "redis", "amqp"],
)
def test_connection_uri_password_is_removed_but_uri_structure_survives(svc, scheme):
    uri = f"{scheme}://service_user:{DATABASE_PASSWORD}@db.internal.example/app"

    output = svc.anonymize(uri)["anonymized_prompt"]

    assert DATABASE_PASSWORD not in output
    assert output.startswith(f"{scheme}://service_user:")
    assert "@db.internal.example/app" in output


def test_dense_env_file_removes_every_secret_and_keeps_variable_names(svc):
    prompt = "\n".join(
        [
            "# synthetic test environment",
            f"OPENAI_API_KEY={OPENAI_KEY}",
            f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}",
            f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY}",
            f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_KEY}",
            f"DATABASE_PASSWORD={DATABASE_PASSWORD}",
        ]
    )

    output = svc.anonymize(prompt, session_id="dense-env")["anonymized_prompt"]

    for secret in [
        OPENAI_KEY,
        ANTHROPIC_KEY,
        AWS_ACCESS_KEY,
        AWS_SECRET_KEY,
        DATABASE_PASSWORD,
    ]:
        assert secret not in output
    for key in [
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "DATABASE_PASSWORD=",
    ]:
        assert key in output
    assert "# synthetic test environment" in output


def test_json_stays_parseable_and_only_values_change(svc):
    original = {
        "provider": "openai",
        "api_key": OPENAI_KEY,
        "database": {
            "username": "service_user",
            "password": DATABASE_PASSWORD,
        },
        "enabled": True,
        "retries": 3,
    }

    output = svc.anonymize(json.dumps(original, separators=(",", ":")))[
        "anonymized_prompt"
    ]
    parsed = json.loads(output)

    assert parsed["provider"] == "openai"
    assert parsed["enabled"] is True
    assert parsed["retries"] == 3
    assert parsed["database"]["username"] == "service_user"
    assert parsed["api_key"] != OPENAI_KEY
    assert parsed["database"]["password"] != DATABASE_PASSWORD


def test_yaml_structure_and_comments_survive(svc):
    prompt = (
        "service:\n"
        "  name: billing\n"
        f"  api_key: {OPENAI_KEY}  # rotate monthly\n"
        f"  password: '{GENERIC_PASSWORD}'\n"
        "  retries: 4\n"
    )

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert "service:\n" in output
    assert "  name: billing\n" in output
    assert "  api_key:" in output
    assert "# rotate monthly" in output
    assert "  password:" in output
    assert "  retries: 4\n" in output
    assert OPENAI_KEY not in output
    assert GENERIC_PASSWORD not in output


def test_same_secret_is_consistent_across_formats_in_one_session(svc):
    session = "secret-cross-format"
    first = svc.anonymize(
        f"OPENAI_API_KEY={OPENAI_KEY}",
        session_id=session,
    )
    second = svc.anonymize(
        json.dumps({"api_key": OPENAI_KEY}),
        session_id=session,
    )

    first_entity = first["entities"][0]
    replacement = first["anonymized_prompt"][
        first_entity["start"]:first_entity["end"]
    ]

    assert replacement in second["anonymized_prompt"]
    assert OPENAI_KEY not in second["anonymized_prompt"]


def test_distinct_secrets_do_not_collapse_to_one_replacement(svc):
    result = svc.anonymize(
        f"primary_token={GITHUB_TOKEN}\nbackup_token={GITLAB_TOKEN}",
        session_id="secret-distinct",
    )
    output = result["anonymized_prompt"]
    replacements = [output[item["start"]:item["end"]] for item in result["entities"]]

    assert GITHUB_TOKEN not in output
    assert GITLAB_TOKEN not in output
    assert len(replacements) == 2
    assert replacements[0] != replacements[1]


@pytest.mark.parametrize(
    "prompt",
    [
        'api_key = os.getenv("API_KEY")',
        "token = ${CI_JOB_TOKEN}",
        "password = <set-at-deploy-time>",
        "const tokenName = 'access_token';",
        "Please rotate the API key without printing it.",
        "-----BEGIN PUBLIC KEY-----\nRkFLRVBVQkxJQ0tFWQ==\n-----END PUBLIC KEY-----",
        "The sk-proj prefix identifies one type of key.",
    ],
)
def test_placeholders_and_secret_related_code_are_not_over_redacted(svc, prompt):
    result = svc.anonymize(prompt)

    assert result["anonymized_prompt"] == prompt
    assert not any(
        item["entity_type"]
        in {"API_KEY", "AUTH_TOKEN", "PRIVATE_KEY", "PASSWORD", "SECRET"}
        for item in result["entities"]
    )


def test_secret_next_to_punctuation_is_fully_removed(svc):
    prompt = f'headers={{"Authorization":"Bearer {JWT}"}}, next=true'

    output = svc.anonymize(prompt)["anonymized_prompt"]

    assert JWT not in output
    assert "headers={" in output
    assert "next=true" in output


def test_private_key_is_removed_as_one_entity(svc):
    result = svc.anonymize(PRIVATE_KEY)

    private_keys = [
        item for item in result["entities"] if item["entity_type"] == "PRIVATE_KEY"
    ]
    assert len(private_keys) == 1
    assert "BEGIN PRIVATE KEY" not in result["anonymized_prompt"]
    assert "END PRIVATE KEY" not in result["anonymized_prompt"]


def test_no_secret_fragment_survives_in_a_mixed_incident_report(svc):
    prompt = (
        "Incident owner John Smith used "
        f"{OPENAI_KEY} from 203.0.113.42. "
        f"Authorization: Bearer {JWT}. "
        f"Database: postgresql://service_user:{DATABASE_PASSWORD}"
        "@db.internal.example/app."
    )

    output = svc.anonymize(prompt, session_id="mixed-incident")["anonymized_prompt"]

    for secret in [OPENAI_KEY, JWT, DATABASE_PASSWORD]:
        assert secret not in output
    assert "John Smith" not in output
    assert "203.0.113.42" not in output
    assert not re.search(r"sk-proj-FAKE|eyJhbGci|FAKE-db-Pass", output)


@pytest.mark.parametrize("label", ["API", "IP", "JSON", "JWT", "URL", "YAML"])
def test_technical_field_labels_are_not_over_redacted(svc, label):
    prompt = f"Inspect the {label} value and preserve the surrounding instructions."

    result = svc.anonymize(prompt)

    assert result["anonymized_prompt"] == prompt
    assert not any(
        item["entity_type"] == "ORGANIZATION" for item in result["entities"]
    )


def test_ip_label_next_to_private_data_is_preserved(svc):
    prompt = "John Smith can be reached at john@example.com from IP 203.0.113.42."
    result = svc.anonymize(prompt)

    assert " from IP " in result["anonymized_prompt"]
    assert "John Smith" not in result["anonymized_prompt"]
    assert "john@example.com" not in result["anonymized_prompt"]
    assert "203.0.113.42" not in result["anonymized_prompt"]
    assert not any(
        item["entity_type"] == "ORGANIZATION"
        and prompt[item["start"] : item["end"]] == "IP"
        for item in result["entities"]
    )
