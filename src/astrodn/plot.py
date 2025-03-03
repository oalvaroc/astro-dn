import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

sns.set_theme()


def plot_image(image: np.ndarray, norm=False, filename=None):
    """Plot FITS image in numpy array."""
    if image.ndim not in [2, 3]:
        raise ValueError(
            f"Image array must be 2D (grayscale) or 3D (RGB) but has {image.dim} dimensions"
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
        plt.savefig(filename, dpi=300)


def plot_loss(loss: list, filename=None):
    sns.lineplot(loss, markers="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)