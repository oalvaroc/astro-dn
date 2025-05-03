"""AstroDnNet model.

Defines the AstroDnNet model, a U-Net-like convolutional neural network
for image denoising tasks. The model supports configurable padding modes
and optional batch normalization.

Example:
    model = AstroDnNet(in_ch=1, padding_mode="reflect", bn="first")
    output = model(input_tensor)
"""

import torch.nn as nn

from .common import DoubleConv2d, DownLayer, UpLayer


class AstroDnNet(nn.Module):
    """A U-Net-like convolutional neural network for image denoising.

    This network follows an encoder-decoder architecture with skip
    connections, supporting configurable padding modes and optional batch
    normalization.
    """

    def __init__(self, in_ch, padding_mode="zeros", bn=None):
        """Initializes the AstroDnNet model.

        Args:
            in_ch (int): Number of input channels.
            padding_mode (str, optional): Padding mode for convolutions. Must
                be a value accepted by `torch.nn.Conv2d`, such as "zeros" and
                "reflect". Defaults to "zeros".
            bn (str, optional): Batch normalization strategy.
                See `DoubleConv2d` for possible values. Default is `None`.
        """
        super().__init__()

        padding = "same" if padding_mode else None
        kwargs = {"padding": padding, "padding_mode": padding_mode, "bn": bn}

        self.conv1 = DoubleConv2d(in_ch, 64, **kwargs)
        self.down1 = DownLayer(64, 128, **kwargs)
        self.down2 = DownLayer(128, 256, **kwargs)
        self.down3 = DownLayer(256, 512, **kwargs)

        self.up1 = UpLayer(512, 256, nearest=True, **kwargs)
        self.up2 = UpLayer(256, 128, nearest=True, **kwargs)
        self.up3 = UpLayer(128, 64, nearest=True, **kwargs)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)

        return self.outconv(x7)
