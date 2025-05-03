"""U-Net components.

Contains reusable convolutional and up/downsampling blocks for building
CNN architectures, particularly U-Net-like models used in image processing
tasks such as denoising.

Classes:
    DoubleConv2d: Two 3x3 convolutions with optional batch normalization.
    DownLayer: Max pooling followed by DoubleConv2d.
    Up: Upsampling via transposed convolution or nearest neighbor with skip
        connection support.
    UpLayer: Combines Up and DoubleConv2d with optional channel adjustment.

Based on: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2d(nn.Module):
    """A double 3x3 convolution block with optional batch normalization.

    Applies two convolutional layers with ReLU activation. Batch
    normalization can be applied to the first conv, second conv, both, or
    disabled entirely.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        mid_ch=None,
        padding="same",
        padding_mode="zeros",
        bn="both",
    ):
        """Initializes module.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            mid_ch (int, optional): Number of intermediate channels. Defaults
                to out_ch if not specified.
            padding (str or int, optional): Padding to use in convolutions.
                Defaults to "same".
            padding_mode (str, optional): Padding mode supported by
                `torch.nn.Conv2d`. Defaults to "zeros".
            bn (str or bool or None, optional): Batch normalization placement.
                Options: "first", "last", "both", True (same as "both"), or
                `None` (disable). Defaults to "both".

        Raises:
            ValueError: If `bn` is not one of the supported options.
        """
        super().__init__()

        if not mid_ch:
            mid_ch = out_ch

        bn_values = ("first", "last", "both", True, None)
        if bn not in bn_values:
            raise ValueError(
                f"batch normalization must be one of {bn_values} but got {bn}"
            )

        modules = [
            nn.Conv2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_ch,
                out_ch,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

        if bn in ("first", None):
            modules.pop(4)
        elif bn in ("last", None):
            modules.pop(1)

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        """Applies the double convolutional block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after convolution and activation.
        """
        return self.conv(x)


class Up(nn.Module):
    """Upsampling block.

    This module upsamples the input and concatenates it with a skip
    connection from the encoder. Uses either transposed convolution or
    nearest neighbor.
    """

    def __init__(self, in_ch, nearest=False):
        """Initializes the upsampling block.

        Args:
            in_ch (int): Number of input channels for the upsampled tensor.
            nearest (bool, optional): If True, uses nearest-neighbor
                upsampling. If False, uses a transposed convolution.
                Defaults to False.
        """
        super().__init__()

        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.up = nn.ConvTranspose2d(
                in_ch, in_ch // 2, kernel_size=2, stride=2
            )

    def forward(self, x1, x2):
        """Upsamples and concatenates input with a skip connection.

        Args:
            x1 (torch.Tensor): Input tensor to upsample.
            x2 (torch.Tensor): Skip connection tensor to concatenate.

        Returns:
            torch.Tensor: Output tensor after upsampling, padding, and
                concatenation.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    """Downsampling block combining max pooling and double convolution."""

    def __init__(self, *args, **kwargs):
        """Initializes the downsampling block.

        Args:
            *args: Positional arguments passed to `DoubleConv2d`.
            **kwargs: Keyword arguments passed to `DoubleConv2d`.
        """
        super().__init__()
        self.conv = DoubleConv2d(*args, **kwargs)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Applies max pooling followed by a double convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after pooling and convolution.
        """
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    """Upsampling block with skip connection and double convolution.

    This layer upsamples the input, concatenates it with a skip connection,
    and applies a double convolution.
    """

    def __init__(self, in_ch, out_ch, *args, nearest=False, **kwargs):
        """Initializes the upsampling and convolutional layer.

        Args:
            in_ch (int): Number of input channels for the upsampled tensor.
            out_ch (int): Number of output channels after convolution.
            *args: Additional positional arguments for `DoubleConv2d`.
            nearest (bool, optional): If True, uses nearest-neighbor
                upsampling. If False, uses transposed convolution.
                Defaults to False.
            **kwargs: Additional keyword arguments for `DoubleConv2d`.
        """
        super().__init__()
        self.up = Up(in_ch, nearest)

        if nearest:
            self.conv = DoubleConv2d(
                in_ch + out_ch, out_ch, in_ch, *args, **kwargs
            )
        else:
            self.conv = DoubleConv2d(in_ch, out_ch, *args, **kwargs)

    def forward(self, x1, x2):
        """Performs upsampling, skip connection, and convolution.

        Args:
            x1 (torch.Tensor): Input tensor to upsample.
            x2 (torch.Tensor): Skip connection tensor to concatenate.

        Returns:
            torch.Tensor: Output tensor after upsampling and convolution.
        """
        y = self.up(x1, x2)
        return self.conv(y)
