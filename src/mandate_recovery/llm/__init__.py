"""The model layer: cached, schema-validated, temperature-0 access.

Nothing here decides anything. Every stage proposes into a deterministic
component that approves, corrects, or refuses — see
`mandate_recovery.agent.validator`. The model's job is inference on ambiguous
evidence, which is the one thing the rules cannot do.
"""

from .cache import DEFAULT_CACHE_DIR, ResponseCache, cache_key
from .client import (
    DEFAULT_MODEL,
    PROVIDER,
    LLMClient,
    LLMCounters,
    LLMFallback,
    StubClient,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ResponseCache",
    "cache_key",
    "DEFAULT_MODEL",
    "PROVIDER",
    "LLMClient",
    "LLMCounters",
    "LLMFallback",
    "StubClient",
]
