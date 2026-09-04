"""The simulator: the private world, and the mechanics acting on it.

Nothing exported from this package may be handed to a policy. See
:class:`mandate_recovery.sim.world.World` for the boundary this side of the
experiment is expected to hold.
"""

from .world import DAYS_IN_MONTH, HOURS_IN_DAY, World

__all__ = ["DAYS_IN_MONTH", "HOURS_IN_DAY", "World"]
