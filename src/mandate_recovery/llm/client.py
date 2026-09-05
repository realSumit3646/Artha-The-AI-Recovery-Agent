"""The model client: temperature 0, a declared schema, and a way to give up.

Every call declares a pydantic response schema and validates the reply against
it. A reply that will not validate after retries raises :class:`LLMFallback`,
which callers catch to fall back to the deterministic path — the model is
never allowed to take an experiment down with it.

Determinism
-----------
Temperature is 0 on every call, and is not configurable. Combined with the
on-disk cache this makes an LLM arm reproducible from stored artifacts, which
invariant 5 requires. ``gemini-2.5-flash`` was verified to return byte-identical
output for a repeated prompt at temperature 0; the flash-lite models were not,
which is why a faster model was not chosen.

The model is **pinned to an exact version**, never an alias like
``gemini-flash-latest``. An alias silently changes the model under a stored
result, and a result you cannot re-derive is not a result.

Networks that intercept TLS
---------------------------
``google-genai`` verifies against certifi, not the operating system's trust
store, so on a machine running TLS-inspecting security software every call
fails with ``CERTIFICATE_VERIFY_FAILED``. That is an environment problem, not
a code one: set ``SSL_CERT_FILE`` to a bundle that includes the intercepting
root. The client does not paper over it, because silently disabling
certificate verification in a payments codebase would be worse than failing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .cache import DEFAULT_CACHE_DIR, ResponseCache, cache_key

__all__ = [
    "DEFAULT_MODEL",
    "PROVIDER",
    "LLMFallback",
    "LLMCounters",
    "LLMClient",
    "StubClient",
]

#: Pinned, not an alias. Verified deterministic at temperature 0.
DEFAULT_MODEL = "gemini-2.5-flash"

PROVIDER = "google-genai"

#: Never configurable. See the module docstring.
TEMPERATURE = 0.0

MAX_RETRIES = 2
BACKOFF_SECONDS = 1.5

ReplyT = TypeVar("ReplyT", bound=BaseModel)


class LLMFallback(RuntimeError):
    """The model could not produce a valid reply. Use the deterministic path.

    Raised after retries are exhausted, on a schema failure, or on a transport
    error. Callers are expected to catch this and continue, not to propagate
    it: an experiment that dies because an API had a bad minute is an
    experiment nobody can run.
    """


@dataclass
class LLMCounters:
    """What the model layer actually did, for the reliability figure."""

    calls_made: int = 0
    cache_hits: int = 0
    schema_failures: int = 0
    transport_failures: int = 0
    fallbacks_triggered: int = 0
    retries: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)

    def reset(self) -> None:
        for key in self.__dict__:
            setattr(self, key, 0)


class LLMClient:
    """Cached, schema-validated, temperature-0 access to one pinned model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        offline: bool = False,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """
        Args:
            offline: serve only from cache. A cache miss raises
                :class:`LLMFallback` rather than reaching the network, which
                is what ``make reproduce`` runs so a reviewer needs no key.
        """
        self.model = model
        self.cache = ResponseCache(cache_dir)
        self.counters = LLMCounters()
        self._offline = offline
        self._max_retries = max_retries
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None

    @property
    def offline(self) -> bool:
        return self._offline

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise LLMFallback(
                    "no GEMINI_API_KEY is set and the response was not cached"
                )
            from google import genai  # imported lazily: offline runs need no SDK

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        schema: Type[ReplyT],
        *,
        system_instruction: str | None = None,
    ) -> ReplyT:
        """One call. Returns a validated ``schema`` instance or raises.

        Raises:
            LLMFallback: on a cache miss while offline, on transport failure
                after retries, or when the reply will not validate.
        """
        key = cache_key(PROVIDER, self.model, prompt, schema.__name__)

        cached = self.cache.get(key)
        if cached is not None:
            self.counters.cache_hits += 1
            try:
                return schema.model_validate_json(cached)
            except ValidationError:
                # A cached entry that no longer fits its schema is stale, not
                # a model failure. Fall through and call again.
                self.counters.schema_failures += 1

        if self._offline:
            self.counters.fallbacks_triggered += 1
            raise LLMFallback(
                f"offline and no cached response for {schema.__name__}; run "
                "the warming script with an API key to populate the cache"
            )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                self.counters.retries += 1
                time.sleep(BACKOFF_SECONDS ** attempt)
            try:
                raw = self._generate(prompt, schema, system_instruction)
            except Exception as error:  # noqa: BLE001 - transport is opaque
                self.counters.transport_failures += 1
                last_error = error
                continue

            self.counters.calls_made += 1
            try:
                reply = schema.model_validate_json(raw)
            except ValidationError as error:
                self.counters.schema_failures += 1
                last_error = error
                continue

            self.cache.put(
                key,
                raw,
                {
                    "provider": PROVIDER,
                    "model": self.model,
                    "schema": schema.__name__,
                    "prompt": prompt,
                },
            )
            return reply

        self.counters.fallbacks_triggered += 1
        raise LLMFallback(
            f"{schema.__name__} could not be produced after "
            f"{self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _generate(
        self, prompt: str, schema: Type[ReplyT], system_instruction: str | None
    ) -> str:
        from google.genai import types

        client = self._ensure_client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=schema,
                system_instruction=system_instruction,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.counters.total_tokens += int(
                (usage.prompt_token_count or 0)
                + (usage.candidates_token_count or 0)
            )
        return response.text

    def stats(self) -> dict[str, Any]:
        return {**self.counters.as_dict(), "cache": self.cache.stats()}


class StubClient:
    """A client that never touches the network. For tests and for failure drills.

    ``replies`` maps schema name to the object to return. Anything not listed
    raises :class:`LLMFallback`, which is how the resilience tests drive every
    stage down its deterministic path.
    """

    def __init__(
        self,
        replies: dict[str, BaseModel] | None = None,
        *,
        always_fail: bool = False,
    ) -> None:
        self.replies = replies or {}
        self.always_fail = always_fail
        self.counters = LLMCounters()
        self.prompts: list[str] = []
        self.model = "stub"

    @property
    def offline(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        schema: Type[ReplyT],
        *,
        system_instruction: str | None = None,
    ) -> ReplyT:
        self.prompts.append(prompt)
        if self.always_fail or schema.__name__ not in self.replies:
            self.counters.fallbacks_triggered += 1
            raise LLMFallback(f"stub has no reply for {schema.__name__}")
        self.counters.calls_made += 1
        return self.replies[schema.__name__]  # type: ignore[return-value]

    def stats(self) -> dict[str, Any]:
        return {**self.counters.as_dict(), "cache": {"entries": 0}}
