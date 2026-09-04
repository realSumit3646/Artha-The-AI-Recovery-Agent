"""Recovery policies, and the boundary that keeps them honest.

Importing this package registers every policy in
:data:`mandate_recovery.policies.base.POLICY_REGISTRY`, which is what the
boundary test walks.
"""

from .base import POLICY_REGISTRY, Policy
from .do_nothing import DoNothingPolicy

__all__ = ["POLICY_REGISTRY", "Policy", "DoNothingPolicy"]
