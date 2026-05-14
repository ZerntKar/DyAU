"""HimaTalk reference implementation.

This package implements a PyTorch research-code version of the method
described in "Pseudo-AU Guided Dyadic Speech-Driven 3D Facial Motion
Generation".
"""

from .model import HimaTalk, HimaTalkConfig

__all__ = ["HimaTalk", "HimaTalkConfig"]
