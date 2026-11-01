"""Compatibility exports for replay diagnostics.

New code should import from :mod:`src.continual.diffusion_data`.
"""

from .diffusion_data import generated_action_diagnostics

__all__ = ["generated_action_diagnostics"]
