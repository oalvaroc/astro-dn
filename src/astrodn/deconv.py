"""Evaluation tools for image denoising models.

Provides utilities to evaluate denoising performance using PSNR and SSIM,
and to visualize results. Supports classical filters like Wiener and
Richardson-Lucy, as well as PyTorch models.
"""

import os

import torch
from scipy.signal import wiener
from skimage.restoration import richardson_lucy
from torch.utils.data import Dataset
from tqdm import tqdm

from . import plot, utils


def evaluate(ds: Dataset, model="wiener", device="cpu", base_dir=""):
    """Evaluates a denoising model on a dataset and saves example outputs.

    Supports classical methods ("wiener", "rl") or a PyTorch model. Computes
    PSNR and SSIM before and after denoising, and saves visualizations for a
    sample image.

    Args:
        ds (Dataset): Dataset returning (noisy, ground truth, raw) image
            triplets.
        model (str or torch.nn.Module, optional): Denoising method. Can be
            "wiener", "rl", or a PyTorch model. Defaults to "wiener".
        device (str, optional): Computation device ("cpu" or "cuda").
            Defaults to "cpu".
        base_dir (str, optional): Directory to save evaluation images.
            Defaults to "".

    Returns:
        dict: Dictionary with mean PSNR and SSIM for noisy and denoised
            images. Keys: "psnr_noisy", "ssim_noisy", "psnr_denoised",
            "ssim_denoised".

    Raises:
        ValueError: If the `model` argument is not supported.
    """
    model_name = model
    if model == "wiener":

        def model_fn(x):
            return torch.from_numpy(
                wiener(x.squeeze().numpy(), (5, 5))
            ).unsqueeze(0)
    elif model == "rl":

        def model_fn(x):
            psf = utils.psf(size=32, nrad=8)
            result = richardson_lucy(
                x.squeeze().numpy(), psf, num_iter=5, clip=False
            )
            return torch.from_numpy(result).unsqueeze(0)
    elif isinstance(model, torch.nn.Module):
        model_name = "cnn"
        model.eval()
        model.to(device)

        def model_fn(x):
            return model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
    else:
        raise ValueError(
            "model should be 'wiener', 'rl' or a torch.nn.Module instance"
        )

    psnr_noisy, ssim_noisy = [], []
    psnr_denoised, ssim_denoised = [], []

    with torch.no_grad():
        for x, _, raw in tqdm(ds):
            x, raw = x.to(device), raw.to(device)

            denoised = model_fn(x)

            psnr_noisy.append(utils.compute_psnr(x, raw))
            ssim_noisy.append(utils.compute_ssim(x, raw))
            psnr_denoised.append(utils.compute_psnr(denoised, raw))
            ssim_denoised.append(utils.compute_ssim(denoised, raw))

        x, y, raw = ds[0]
        x, y, raw = (x.to(device), y.to(device), raw.to(device))

        denoised = model_fn(x)

        plot.plot_image(
            x[0],
            norm=True,
            filename=os.path.join(base_dir, f"eval-x-{model_name}.png"),
            dpi=300,
        )
        plot.plot_image(
            y[0],
            norm=True,
            filename=os.path.join(base_dir, f"eval-y-{model_name}.png"),
            dpi=300,
        )
        plot.plot_image(
            denoised[0],
            norm=True,
            filename=os.path.join(base_dir, f"eval-denoised-{model_name}.png"),
            dpi=300,
        )

    return {
        "psnr_noisy": sum(psnr_noisy) / len(psnr_noisy),
        "ssim_noisy": sum(ssim_noisy) / len(ssim_noisy),
        "psnr_denoised": sum(psnr_denoised) / len(psnr_denoised),
        "ssim_denoised": sum(ssim_denoised) / len(ssim_denoised),
    }
