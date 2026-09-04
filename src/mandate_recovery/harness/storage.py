"""Where experiment results go, and how they are made reproducible.

Every experiment writes one directory::

    results/<experiment_id>/
        config.json            the full run config, git SHA, simulator hash
        raw/episodes.parquet   one row per mandate per arm
        raw/decisions.parquet  every decision made, the audit trail
        metrics.json           computed summary

The rule that matters is the refusal: :func:`write_experiment` will not
overwrite an existing ``experiment_id`` unless told to in as many words.
Results that can be silently replaced are results nobody can cite, because the
directory a claim points at may no longer hold what produced it.

``config.json`` carries the git SHA and ``SIMULATOR_HASH`` alongside the run
parameters, so a stored result says which world it came from. If the simulator
hash in a stored config does not match the current one, that result predates a
change to the world and cannot be compared with anything produced after it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ..sim.freeze import SIMULATOR_HASH

__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "ExperimentExistsError",
    "experiment_directory",
    "git_sha",
    "write_experiment",
    "read_experiment",
]

DEFAULT_RESULTS_ROOT = Path("results")


class ExperimentExistsError(RuntimeError):
    """Raised rather than overwriting a stored experiment."""


def git_sha(repo_root: Path | None = None) -> str:
    """Current commit, or ``"unknown"`` outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root or Path(__file__).resolve().parents[3]),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def experiment_directory(
    experiment_id: str, results_root: Path | None = None
) -> Path:
    return Path(results_root or DEFAULT_RESULTS_ROOT) / experiment_id


def write_experiment(
    experiment_id: str,
    config: Mapping[str, Any],
    episodes: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: Mapping[str, Any],
    *,
    results_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one experiment's full output. Refuses to clobber by default.

    Returns the directory written.
    """
    directory = experiment_directory(experiment_id, results_root)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ExperimentExistsError(
            f"experiment {experiment_id!r} already exists at {directory}. "
            "Pass overwrite=True only if you mean to discard the stored run; "
            "anything citing it will then be pointing at different numbers."
        )

    (directory / "raw").mkdir(parents=True, exist_ok=True)

    stamped = {
        "experiment_id": experiment_id,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "simulator_hash": SIMULATOR_HASH,
        **dict(config),
    }
    (directory / "config.json").write_text(
        json.dumps(stamped, indent=2, default=str), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, default=str), encoding="utf-8"
    )

    episodes.to_parquet(directory / "raw" / "episodes.parquet", index=False)
    decisions.to_parquet(directory / "raw" / "decisions.parquet", index=False)
    return directory


def read_experiment(
    experiment_id: str, results_root: Path | None = None
) -> dict[str, Any]:
    """Load a stored experiment back: config, metrics, episodes, decisions."""
    directory = experiment_directory(experiment_id, results_root)
    if not directory.exists():
        raise FileNotFoundError(f"no experiment at {directory}")

    return {
        "config": json.loads((directory / "config.json").read_text("utf-8")),
        "metrics": json.loads((directory / "metrics.json").read_text("utf-8")),
        "episodes": pd.read_parquet(directory / "raw" / "episodes.parquet"),
        "decisions": pd.read_parquet(directory / "raw" / "decisions.parquet"),
    }


def records_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Rows to a DataFrame, keeping an empty result well-typed."""
    rows = list(records)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
