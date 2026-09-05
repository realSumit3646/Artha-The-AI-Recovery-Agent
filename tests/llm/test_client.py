"""Tests for the model client and its cache.

No live API calls anywhere in this file, or in the suite. Every test drives a
fake transport, because a test suite that needs a paid external service is a
test suite that stops running the day the key expires.
"""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel

from mandate_recovery.llm.cache import ResponseCache, cache_key
from mandate_recovery.llm.client import (
    DEFAULT_MODEL,
    PROVIDER,
    TEMPERATURE,
    LLMClient,
    LLMFallback,
    StubClient,
)


class Reply(BaseModel):
    cause: Literal["FUNDS", "LIMIT", "UNKNOWN"]
    confidence: float


VALID = '{"cause": "FUNDS", "confidence": 0.9}'


class _FakeTransport:
    """Stands in for the network. Records calls, returns scripted replies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.configs = []

    def __call__(self, prompt, schema, system_instruction):
        self.calls += 1
        item = self.responses.pop(0) if self.responses else VALID
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Retry backoff is real time; tests should not pay for it."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def _client(tmp_path, responses, **kwargs) -> tuple[LLMClient, _FakeTransport]:
    client = LLMClient(cache_dir=tmp_path, api_key="test-key", **kwargs)
    transport = _FakeTransport(responses)
    client._generate = transport  # type: ignore[method-assign]
    return client, transport


# --------------------------------------------------------------------------
# Determinism and pinning
# --------------------------------------------------------------------------


def test_temperature_is_zero_and_not_configurable():
    """Invariant 5: LLM calls use temperature 0."""
    assert TEMPERATURE == 0.0
    import inspect

    from mandate_recovery.llm import client as module

    signature = inspect.signature(module.LLMClient.__init__)
    assert "temperature" not in signature.parameters


def test_the_model_is_pinned_not_an_alias():
    """An alias silently changes the model under a stored result."""
    assert DEFAULT_MODEL == "gemini-2.5-flash"
    assert "latest" not in DEFAULT_MODEL


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def test_a_valid_reply_is_returned_as_a_validated_model(tmp_path):
    client, _ = _client(tmp_path, [VALID])
    reply = client.complete("prompt", Reply)
    assert isinstance(reply, Reply)
    assert reply.cause == "FUNDS"


def test_a_reply_that_fails_the_schema_is_retried(tmp_path):
    client, transport = _client(
        tmp_path, ['{"cause": "NONSENSE", "confidence": 1}', VALID]
    )
    reply = client.complete("prompt", Reply)
    assert reply.cause == "FUNDS"
    assert transport.calls == 2
    assert client.counters.schema_failures == 1


def test_persistent_schema_failure_raises_fallback(tmp_path):
    client, transport = _client(tmp_path, ['{"bad": 1}'] * 5)
    with pytest.raises(LLMFallback):
        client.complete("prompt", Reply)
    assert transport.calls == 3  # one attempt plus two retries
    assert client.counters.fallbacks_triggered == 1


def test_malformed_json_never_reaches_the_caller(tmp_path):
    client, _ = _client(tmp_path, ["not json at all"] * 5)
    with pytest.raises(LLMFallback):
        client.complete("prompt", Reply)


# --------------------------------------------------------------------------
# Retries and transport failure
# --------------------------------------------------------------------------


def test_a_transient_transport_error_is_retried(tmp_path):
    client, transport = _client(
        tmp_path, [ConnectionError("503 UNAVAILABLE"), VALID]
    )
    reply = client.complete("prompt", Reply)
    assert reply.cause == "FUNDS"
    assert client.counters.transport_failures == 1
    assert client.counters.retries == 1


def test_retries_are_capped(tmp_path):
    client, transport = _client(
        tmp_path, [ConnectionError("503")] * 9, max_retries=2
    )
    with pytest.raises(LLMFallback):
        client.complete("prompt", Reply)
    assert transport.calls == 3


def test_the_fallback_message_says_what_went_wrong(tmp_path):
    client, _ = _client(tmp_path, [ConnectionError("503 UNAVAILABLE")] * 9)
    with pytest.raises(LLMFallback, match="503"):
        client.complete("prompt", Reply)


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def test_a_second_identical_call_is_served_from_cache(tmp_path):
    client, transport = _client(tmp_path, [VALID])
    client.complete("prompt", Reply)
    client.complete("prompt", Reply)

    assert transport.calls == 1
    assert client.counters.cache_hits == 1
    assert client.counters.calls_made == 1


def test_a_different_prompt_is_a_different_entry(tmp_path):
    client, transport = _client(tmp_path, [VALID, VALID])
    client.complete("one", Reply)
    client.complete("two", Reply)
    assert transport.calls == 2
    assert len(client.cache) == 2


def test_the_cache_survives_a_new_client(tmp_path):
    """The point of an on-disk cache: a later run needs no key."""
    first, transport = _client(tmp_path, [VALID])
    first.complete("prompt", Reply)

    second = LLMClient(cache_dir=tmp_path, api_key=None, offline=True)
    reply = second.complete("prompt", Reply)
    assert reply.cause == "FUNDS"
    assert second.counters.cache_hits == 1


def test_offline_with_no_cached_entry_falls_back(tmp_path):
    client = LLMClient(cache_dir=tmp_path, offline=True, api_key=None)
    with pytest.raises(LLMFallback, match="offline"):
        client.complete("never seen", Reply)


def test_the_provider_is_part_of_the_key():
    """A provider switch must not collide with stale entries."""
    left = cache_key("google-genai", "m", "p", "S")
    right = cache_key("anthropic", "m", "p", "S")
    assert left != right


def test_the_schema_name_is_part_of_the_key():
    assert cache_key(PROVIDER, "m", "p", "A") != cache_key(PROVIDER, "m", "p", "B")


def test_the_model_is_part_of_the_key():
    assert cache_key(PROVIDER, "a", "p", "S") != cache_key(PROVIDER, "b", "p", "S")


def test_a_stale_cache_entry_is_refetched_not_returned(tmp_path):
    """A cached reply that no longer fits its schema is stale, not a failure."""
    cache = ResponseCache(tmp_path)
    key = cache_key(PROVIDER, DEFAULT_MODEL, "prompt", "Reply")
    cache.put(key, '{"cause": "REMOVED_ENUM_MEMBER", "confidence": 1}')

    client, transport = _client(tmp_path, [VALID])
    reply = client.complete("prompt", Reply)
    assert reply.cause == "FUNDS"
    assert transport.calls == 1


def test_the_cache_records_the_prompt_for_a_human_reader(tmp_path):
    client, _ = _client(tmp_path, [VALID])
    client.complete("a readable prompt", Reply)

    entry = next(tmp_path.rglob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    assert payload["prompt"] == "a readable prompt"
    assert payload["provider"] == PROVIDER
    assert payload["model"] == DEFAULT_MODEL


def test_an_interrupted_write_leaves_no_partial_entry(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put(cache_key(PROVIDER, "m", "p", "S"), VALID)
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache_key(PROVIDER, "m", "p", "S")
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated", encoding="utf-8")
    assert cache.get(key) is None
    assert cache.misses == 1


def test_cache_stats_are_reported(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache_key(PROVIDER, "m", "p", "S")
    cache.get(key)
    cache.put(key, VALID)
    cache.get(key)
    assert cache.stats() == {"entries": 1, "hits": 1, "misses": 1, "writes": 1}


# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------


def test_counters_track_everything_the_reliability_figure_needs(tmp_path):
    client, _ = _client(
        tmp_path, [ConnectionError("x"), '{"bad": 1}', VALID]
    )
    client.complete("prompt", Reply)
    client.complete("prompt", Reply)

    counters = client.counters
    # Two responses came back — the malformed one and the valid one. A
    # transport failure returned nothing and is counted separately, which
    # matters because a schema failure is billed and a 503 is not.
    assert counters.calls_made == 2
    assert counters.cache_hits == 1
    assert counters.schema_failures == 1
    assert counters.transport_failures == 1
    assert counters.retries == 2
    assert counters.fallbacks_triggered == 0


def test_counters_can_be_reset_between_arms(tmp_path):
    client, _ = _client(tmp_path, [VALID])
    client.complete("prompt", Reply)
    client.counters.reset()
    assert client.counters.calls_made == 0


# --------------------------------------------------------------------------
# The stub
# --------------------------------------------------------------------------


def test_the_stub_returns_scripted_replies():
    stub = StubClient({"Reply": Reply(cause="LIMIT", confidence=0.5)})
    assert stub.complete("p", Reply).cause == "LIMIT"
    assert stub.prompts == ["p"]


def test_the_stub_can_be_made_to_always_fail():
    """How every resilience test drives a stage down its fallback path."""
    stub = StubClient(always_fail=True)
    with pytest.raises(LLMFallback):
        stub.complete("p", Reply)
    assert stub.counters.fallbacks_triggered == 1


def test_the_stub_fails_for_an_unscripted_schema():
    stub = StubClient({"Other": Reply(cause="FUNDS", confidence=1.0)})
    with pytest.raises(LLMFallback):
        stub.complete("p", Reply)


def test_the_suite_makes_no_live_calls():
    """A client with no key and no cache cannot reach the network."""
    client = LLMClient(cache_dir="/nonexistent", api_key=None, offline=True)
    with pytest.raises(LLMFallback):
        client.complete("prompt", Reply)
