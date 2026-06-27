"""DyAU reference implementation.

This package implements a PyTorch research-code version of the method
described in "DyAU: Interaction-Aware Regional Priors for Dyadic
Speech-Driven 3D Facial Motion Generation".
"""

from .model import DyAU, DyAUConfig

__all__ = ["DyAU", "DyAUConfig"]
