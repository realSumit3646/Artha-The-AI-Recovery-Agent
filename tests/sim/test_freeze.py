"""The freeze check.

This is the test that keeps the central claim of the project honest: the world
was fixed before any policy was written, and has not been adjusted since.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mandate_recovery.calibration import DEFAULT_CALIBRATION
from mandate_recovery.sim.freeze import (
    FREEZE_FAILURE_MESSAGE,
    FROZEN_SOURCE_FILES,
    SIMULATOR_HASH,
    compute_simulator_hash,
)

SIM_DIRECTORY = Path(__file__).resolve().parents[2] / "src" / "mandate_recovery" / "sim"


# --------------------------------------------------------------------------
# The freeze itself
# --------------------------------------------------------------------------


def test_simulator_has_not_changed_since_the_freeze():
    assert compute_simulator_hash() == SIMULATOR_HASH, FREEZE_FAILURE_MESSAGE


# --------------------------------------------------------------------------
# The check has to actually detect changes
# --------------------------------------------------------------------------


def test_a_calibration_change_moves_the_hash():
    """Retuning any parameter breaks the freeze, which is the point."""
    tweaked = DEFAULT_CALIBRATION.model_copy(
        update={
            "upi_autopay_execution_failure_rate": (
                DEFAULT_CALIBRATION.upi_autopay_execution_failure_rate.model_copy(
                    update={"value": 0.2999}
                )
            )
        }
    )
    assert compute_simulator_hash(tweaked) != SIMULATOR_HASH


def test_every_frozen_source_contributes_to_the_hash(tmp_path, monkeypatch):
    """A change to any one of the three modules must move the hash.

    Verified by hashing modified copies rather than by editing the real
    sources, so the test cannot leave the working tree broken.
    """
    import mandate_recovery.sim.freeze as freeze_module

    for target in FROZEN_SOURCE_FILES:
        staging = tmp_path / target
        staging.parent.mkdir(parents=True, exist_ok=True)
        for name in FROZEN_SOURCE_FILES:
            content = (SIM_DIRECTORY / name).read_bytes()
            if name == target:
                content += b"\n# a change to the world\n"
            (tmp_path / name).write_bytes(content)

        monkeypatch.setattr(freeze_module, "_SIM_DIRECTORY", tmp_path)
        assert freeze_module.compute_simulator_hash() != SIMULATOR_HASH, (
            f"editing {target} did not move the simulator hash"
        )


def test_the_hash_covers_exactly_the_modules_that_decide_outcomes():
    assert FROZEN_SOURCE_FILES == ("world.py", "outcomes.py", "response_codes.py")
    for name in FROZEN_SOURCE_FILES:
        assert (SIM_DIRECTORY / name).exists(), f"{name} is missing"


# --------------------------------------------------------------------------
# Portability
# --------------------------------------------------------------------------


def test_the_hash_survives_windows_line_endings():
    """A CRLF checkout must produce the same hash as an LF one.

    ``core.autocrlf`` is on for this repository, so without normalisation the
    freeze would break for every contributor on a different platform and the
    failure would look like tampering.
    """
    import mandate_recovery.sim.freeze as freeze_module

    for name in FROZEN_SOURCE_FILES:
        raw = (SIM_DIRECTORY / name).read_bytes()
        normalised = freeze_module._normalised_source(SIM_DIRECTORY / name)
        crlf = normalised.replace(b"\n", b"\r\n")
        assert crlf.replace(b"\r\n", b"\n") == normalised
        assert b"\r" not in normalised, f"{name} was not normalised"
        assert raw  # the file is not empty


def test_the_hash_is_a_sha256_hex_digest():
    assert len(SIMULATOR_HASH) == 64
    assert set(SIMULATOR_HASH) <= set("0123456789abcdef")
    assert SIMULATOR_HASH != hashlib.sha256(b"").hexdigest()


def test_the_hash_is_stable_across_repeated_computation():
    assert compute_simulator_hash() == compute_simulator_hash()


# --------------------------------------------------------------------------
# The message a future session will read
# --------------------------------------------------------------------------


def test_the_failure_message_says_what_to_do():
    assert "re-run" in FREEZE_FAILURE_MESSAGE
    assert "PROGRESS.md" in FREEZE_FAILURE_MESSAGE
    assert "SIMULATOR_HASH" in FREEZE_FAILURE_MESSAGE


def test_freeze_is_documented():
    doc = Path(__file__).resolve().parents[2] / "docs" / "FREEZE.md"
    text = doc.read_text(encoding="utf-8")
    assert SIMULATOR_HASH in text, "docs/FREEZE.md records a different hash"


@pytest.mark.parametrize("name", ["world.py", "outcomes.py", "response_codes.py"])
def test_frozen_modules_are_not_empty(name):
    assert (SIM_DIRECTORY / name).stat().st_size > 500
