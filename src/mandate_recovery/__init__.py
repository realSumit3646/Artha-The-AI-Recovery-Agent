"""Artha — a simulation harness for UPI Autopay recurring-payment failures.

The core loop is simulator -> policy -> metrics. The simulator holds latent
state; policies see only an Observation.
"""

__version__ = "0.1.0"
