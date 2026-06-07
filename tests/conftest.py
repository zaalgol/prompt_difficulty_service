"""Shared test fixtures.

The /anonymize session vault is backed by Redis. Tests must not require a real
server, so we transparently swap the service's Redis client for an in-process
fakeredis instance shared across the whole session. Keys are flushed after each
test module so a session_id used in one module cannot leak into another and so
re-running the suite has no side effects.
"""
import pytest

import app.presidio_service.service as anon_service


@pytest.fixture(scope="session", autouse=True)
def fake_redis():
    """Point AnonymizerService at one shared in-process fakeredis for the session."""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeStrictRedis(decode_responses=True)

    original_factory = anon_service._make_redis_client
    anon_service._make_redis_client = lambda *args, **kwargs: client
    try:
        yield client
    finally:
        anon_service._make_redis_client = original_factory
        client.flushall()


@pytest.fixture(autouse=True, scope="module")
def _flush_vault_between_modules(fake_redis):
    """Delete all vault keys after each test module to keep modules isolated and
    reruns side-effect-free. Module scope (not function) so a module-scoped
    service/conversation fixture can build up state across its own tests."""
    yield
    fake_redis.flushall()
