"""The evaluation harness: paired experiment execution and result storage."""

from .runner import (
    ArmResult,
    DecisionRecord,
    EpisodeRecord,
    ExperimentConfig,
    build_observation,
    build_world_and_mandates,
    run_experiment,
)
from .metrics import (
    compare_arms,
    compute_metrics,
    compute_metrics_by_arm,
    summarise_comparison,
)
from .storage import (
    ExperimentExistsError,
    experiment_directory,
    read_experiment,
    write_experiment,
)

__all__ = [
    "ArmResult",
    "DecisionRecord",
    "EpisodeRecord",
    "ExperimentConfig",
    "build_observation",
    "build_world_and_mandates",
    "run_experiment",
    "compare_arms",
    "compute_metrics",
    "compute_metrics_by_arm",
    "summarise_comparison",
    "ExperimentExistsError",
    "experiment_directory",
    "read_experiment",
    "write_experiment",
]
