"""On-disk response cache. The thing that makes this project reproducible.

Every model call is keyed on a SHA256 of ``(provider, model, prompt, schema)``
and stored as JSON on disk. The cache directory is **committed to git**, which
means a reviewer can clone this repository with no API key, run
``make reproduce``, and get the same numbers that are in the README.

That is not a convenience. An experiment whose results depend on a paid
external service that returns different text next month is not reproducible,
and the honest fix is to ship the responses alongside the code.

Why the provider is in the key
------------------------------
This project started against a different provider. Without the provider in the
key, a switch would silently collide with stale entries and the run would look
reproducible while quietly mixing two models' answers.

On prompt design
----------------
The cache only earns its keep if prompts repeat. They repeat because the
prompt builders render a **canonical, bucketed** view of an observation rather
than its raw numbers -- see ``mandate_recovery.llm.diagnosis``. Thousands of
failures collapse onto a few hundred distinct prompts, so a full experiment
costs a few hundred calls once and nothing thereafter.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping

__all__ = ["DEFAULT_CACHE_DIR", "ResponseCache", "cache_key"]

#: Committed to git on purpose. See the module docstring.
DEFAULT_CACHE_DIR = Path("llm_cache")


def cache_key(provider: str, model: str, prompt: str, schema_name: str) -> str:
    """Stable key for one call. Order and separators are part of the key."""
    digest = hashlib.sha256()
    for part in (provider, model, schema_name, prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class ResponseCache:
    """A directory of JSON files, one per distinct call.

    One file per key rather than a single index, so concurrent warming does
    not corrupt anything and a diff shows exactly which prompts changed.
    """

    def __init__(self, directory: Path | str = DEFAULT_CACHE_DIR) -> None:
        self._directory = Path(directory)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, key: str) -> Path:
        # Shard by the first two characters: a flat directory of thousands of
        # files is unpleasant in git and slow on Windows.
        return self._directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return payload.get("response")

    def put(self, key: str, response: str, metadata: Mapping[str, Any] | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"response": response, **(dict(metadata) if metadata else {})}
        # Write via a temporary file so an interrupted run cannot leave a
        # half-written entry that later reads as a valid cached response.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
        with self._lock:
            self.writes += 1

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def __len__(self) -> int:
        if not self._directory.exists():
            return 0
        return sum(1 for _ in self._directory.rglob("*.json"))

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def reset_counters(self) -> None:
        self.hits = self.misses = self.writes = 0
