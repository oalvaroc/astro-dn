"""Denoising dataset utilities.

Provides dataset classes and tools for loading and preparing image data
stored as .npy files, primarily for denoising applications.
"""

from .npy import NPYDataset

__all__ = [NPYDataset]
