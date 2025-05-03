"""Plotting utilities for denoising experiments.

Provides functions to visualize training metrics and image data,
supporting both real-time inspection and high-resolution file export.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

sns.set_theme()


def plot_image(image: np.ndarray, norm=True, filename=None, dpi=96):
    """Plots a 2D or 3D image array with optional normalization.

    Displays or saves a grayscale or RGB image, optionally normalized using
    ZScale and linear stretch (useful for astronomical images).

    Args:
        image (np.ndarray): Image array to plot. Must be 2D (grayscale) or
            3D (RGB).
        norm (bool, optional): Whether to apply normalization using
            `ImageNormalize` and `ZScaleInterval`. Defaults to True.
        filename (str, optional): Path to save the image. If None, the image
            is not saved. Defaults to None.
        dpi (int, optional): DPI for saved image if `filename` is provided.
            Defaults to 96.

    Raises:
        ValueError: If `image` is not 2D or 3D.
    """
    if image.ndim not in [2, 3]:
        raise ValueError(
            f"Image array must be 2D (grayscale) or 3D (RGB) but has {image.dim} dimensions"  # noqa: E501
        )

    plt.figure(figsize=(8, 6))
    if norm:
        norm_filter = ImageNormalize(
            image, interval=ZScaleInterval(), stretch=LinearStretch()
        )
        plt.imshow(image, cmap="gray", norm=norm_filter)
    else:
        plt.imshow(image, cmap="gray")

    plt.colorbar()
    plt.grid(visible=False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi)
    plt.close()


def plot(
    data: list,
    filename=None,
    dpi=96,
    title="Loss curve",
    xlabel="Epoch",
    ylabel="Loss",
):
    """Plots a line graph from a list of values.

    Typically used to visualize loss over training epochs.

    Args:
        data (list): List of scalar values to plot.
        filename (str, optional): Path to save the plot. If None, the plot is
            not saved. Defaults to None.
        dpi (int, optional): DPI for saved image if `filename` is provided.
            Defaults to 96.
        title (str, optional): Title of the plot. Defaults to "Loss curve".
        xlabel (str, optional): Label for the x-axis. Defaults to "Epoch".
        ylabel (str, optional): Label for the y-axis. Defaults to "Loss".
    """
    sns.lineplot(data, linestyle="-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=dpi)
    plt.close()
