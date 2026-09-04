"""The simulator freeze.

The world was fixed before any policy was written. This module is what makes
that claim checkable rather than merely asserted: it hashes the calibration
and the source of the three modules that decide what happens to a payment, and
pins the result as :data:`SIMULATOR_HASH`.

If the hash moves, the world moved, and every number measured against the old
world is stale. ``tests/sim/test_freeze.py`` fails loudly when that happens.

Why this exists
---------------
A simulator whose parameters can be adjusted after seeing how a policy scores
is not an experiment; it is a search for flattering settings. Freezing removes
the temptation by making any change to the world visible in a diff and fatal
to the test suite. The honest recovery from a genuine simulator bug is to fix
it, re-run **every** arm from scratch, record the re-baseline in
``PROGRESS.md``, and only then update the constant here.

What is covered
---------------
The serialised :class:`CalibrationSet` and the normalised source text of
``world.py``, ``outcomes.py`` and ``response_codes.py``. Line endings are
normalised to ``\\n`` before hashing, so a checkout on Windows produces the
same hash as one on Linux.

This module is deliberately not hashed: it holds the hash, and hashing itself
is circular. See ``docs/FREEZE.md`` for what else sits outside the boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..calibration import DEFAULT_CALIBRATION, CalibrationSet

__all__ = [
    "FROZEN_SOURCE_FILES",
    "SIMULATOR_HASH",
    "compute_simulator_hash",
    "FREEZE_FAILURE_MESSAGE",
]

_SIM_DIRECTORY: Final = Path(__file__).resolve().parent

#: The modules that decide what happens to a payment attempt. Hashed in this
#: order; the order is part of the hash.
FROZEN_SOURCE_FILES: Final = ("world.py", "outcomes.py", "response_codes.py")

FREEZE_FAILURE_MESSAGE: Final = (
    "Simulator changed after freeze. If this was intentional, ALL experiment "
    "arms must be re-run from scratch and the re-baseline recorded in "
    "PROGRESS.md. Update SIMULATOR_HASH only after doing so."
)


def _normalised_source(path: Path) -> bytes:
    """File bytes with line endings normalised, so the hash is portable."""
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def compute_simulator_hash(
    calibration: CalibrationSet = DEFAULT_CALIBRATION,
) -> str:
    """SHA256 over the calibration and the frozen simulator sources."""
    digest = hashlib.sha256()
    digest.update(b"calibration\0")
    digest.update(calibration.model_dump_json().encode("utf-8"))
    for name in FROZEN_SOURCE_FILES:
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalised_source(_SIM_DIRECTORY / name))
    return digest.hexdigest()


#: The frozen world, as of the freeze recorded in ``docs/FREEZE.md``.
#:
#: Do not update this to make a test pass. Updating it is the last step of a
#: deliberate re-baseline, never the first.
SIMULATOR_HASH: Final = (
    "fd5a8fed4eaf5a6d719e9470a2978f93dfd2dccfe0fd2e59de65801b9b31b193"
)
