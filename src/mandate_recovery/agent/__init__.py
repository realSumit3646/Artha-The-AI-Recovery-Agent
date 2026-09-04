"""The deterministic agent: scheduling, compliance validation, audit.

Nothing in this package consults a language model. Timing is constrained
optimisation and compliance is a gate — both are arithmetic, and both must be
reproducible from a stored config. The model layer, when it arrives, proposes
into these components rather than replacing them.
"""

from .audit import AuditEntry, AuditLog, observation_fingerprint
from .validator import (
    Budget,
    ComplianceLimits,
    ValidationResult,
    Validator,
    validate,
)
from .scheduler import (
    RetrySlot,
    SchedulerConstraints,
    infer_bank_tier,
    next_retry_slot,
)

__all__ = [
    "AuditEntry",
    "AuditLog",
    "observation_fingerprint",
    "Budget",
    "ComplianceLimits",
    "ValidationResult",
    "Validator",
    "validate",
    "RetrySlot",
    "SchedulerConstraints",
    "infer_bank_tier",
    "next_retry_slot",
]
