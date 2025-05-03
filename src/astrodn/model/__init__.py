"""Package implementing AstroDnNet.

This package provides the AstroDnNet architecture and its supporting
building blocks. AstroDnNet is a configurable U-Net variant designed for
astronomical image denoising tasks.

Modules:
    net: Defines the `AstroDnNet` model.
    common: Contains reusable layers used in AstroDnNet, including
        `DoubleConv2d`, `DownLayer`, `Up`, and `UpLayer`.
"""

from .net import AstroDnNet

__all__ = [AstroDnNet]