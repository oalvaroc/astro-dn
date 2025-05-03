"""Image processing utilities for denoising and evaluation.

Provides functions for patch-based model inference, image corruption,
visualization, FITS loading, PSNR/SSIM computation, and loss saving.
"""

import csv
import os

import astropy.io.fits as fits
import numpy as np
import scipy
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def loadfits(file):
    """Loads a FITS image file as a NumPy float32 array.

    Args:
        file (str): Path to the FITS file.

    Returns:
        np.ndarray: Image data as a float32 NumPy array.
    """
    return fits.getdata(file).astype(np.float32)


def split_into_patches(
    image: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """Splits an image into overlapping patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels,
            height, width).
        patch_size (int): The size of the extracted patches.
        stride (int): The stride between patches when splitting.

    Returns:
        torch.Tensor: Extracted patches of shape
                      (batch, num_patches, channels, patch_size, patch_size).
    """  # noqa: E501
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    patches = unfold(
        image
    )  # Shape: (batch, channels * patch_size^2, num_patches)

    batch_size, _, num_patches = patches.shape
    patches = patches.view(batch_size, -1, patch_size, patch_size, num_patches)
    patches = patches.permute(
        0, 4, 1, 2, 3
    )  # (batch, num_patches, channels, height, width)

    return patches


def reconstruct_image(
    processed_patches: torch.Tensor,
    original_image_shape: tuple,
    patch_size: int,
    output_patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Reconstructs the image from processed patches.

    Args:
        processed_patches (torch.Tensor): Model-processed patches of shape
            (batch, num_patches, channels, output_patch_size, output_patch_size).
        original_image_shape (tuple): The shape (height, width) of the original image.
        patch_size (int): The size of the original extracted patches.
        output_patch_size (int): The size of the patches after processing.
        stride (int): The stride used during patch extraction.

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (batch, channels, height', width').
    """
    batch_size, num_patches, channels, _, _ = processed_patches.shape

    # Reshape for folding
    processed_patches = processed_patches.permute(0, 2, 3, 4, 1).contiguous()
    processed_patches = processed_patches.view(
        batch_size,
        channels * output_patch_size * output_patch_size,
        num_patches,
    )

    # Compute expected output size based on stride
    original_height, original_width = original_image_shape
    final_height = (
        (original_height - patch_size) // stride
    ) * stride + output_patch_size
    final_width = (
        (original_width - patch_size) // stride
    ) * stride + output_patch_size

    # Fold back to reconstruct image
    fold = nn.Fold(
        output_size=(final_height, final_width),
        kernel_size=output_patch_size,
        stride=stride,
    )

    # Create normalization tensor to handle overlapping contributions
    normalizer = fold(torch.ones_like(processed_patches))
    reconstructed_image = fold(processed_patches) / normalizer

    return reconstructed_image[:, :, :final_height, :final_width]


def process_image_with_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int,
    output_patch_size: int,
    stride: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Splits an image into patches.

    Args:
        model (torch.nn.Module): The model that processes each patch.
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        patch_size (int): The size of the extracted patches.
        output_patch_size (int): The size of the patches after model processing.
        stride (int): The stride between patches when splitting.
        batch_size (int, optional): The number of patches to process per batch.
                                    Defaults to 8.

    Returns:
        torch.Tensor: The reconstructed output image of shape (batch, channels, height', width').
    """
    # Extract patches
    patches = split_into_patches(image, patch_size, stride)

    batch_size_total, num_patches, channels, _, _ = patches.shape
    processed_patches = []

    # Process patches in batches
    for start in range(0, num_patches, batch_size):
        batch_patches = patches[:, start : start + batch_size]  # Select batch
        batch_size_current, num_batch_patches, channels, height, width = (
            batch_patches.shape
        )
        batch_patches = batch_patches.view(
            batch_size_current * num_batch_patches, channels, height, width
        )  # Flatten batch

        with torch.no_grad():
            batch_output = model(batch_patches)  # Model inference

        batch_output = batch_output.view(
            batch_size_current,
            num_batch_patches,
            channels,
            output_patch_size,
            output_patch_size,
        )
        processed_patches.append(batch_output)

    # Concatenate all processed patches
    processed_patches = torch.cat(processed_patches, dim=1)

    # Reconstruct image
    return reconstruct_image(
        processed_patches,
        image.shape[-2:],
        patch_size,
        output_patch_size,
        stride,
    )


def save_losses(losses: list, filename: str):
    """Saves a list of scalar loss values to a CSV file.

    Args:
        losses (list): List of scalar loss values.
        filename (str): Output file path. Must end with ".csv".

    Raises:
        ValueError: If the filename does not end with ".csv".
    """
    _, ext = os.path.splitext(filename)
    if ext != ".csv":
        raise ValueError("File name should end in '.csv'")

    with open(filename, "w+") as f:
        writer = csv.writer(f)
        writer.writerows([[loss] for loss in losses])


def psf(size=32, nrad=5):
    """Generates a circular point spread function using a Bessel profile.

    The PSF is computed using the squared normalized first-order Bessel
    function of the first kind.

    Args:
        size (int, optional): Size (width and height) of the square PSF
            array. Must be > 0. Defaults to 32.
        nrad (int, optional): Radial frequency scaling factor. Controls the
            spread of the PSF. Defaults to 5.

    Returns:
        np.ndarray: 2D array representing the normalized PSF.
    """
    assert size > 0
    arange = np.linspace(-1, 1, size)
    x, y = np.meshgrid(arange, arange)
    r = np.sqrt(x**2 + y**2) * np.pi * nrad
    psf = (scipy.special.j1(r) / r) ** 2
    psf = psf / psf.sum()
    return psf


def corrupt_np(data: np.ndarray, mode: str = "poisson", size=64, sigma=0.01):
    """Applies synthetic corruption to a NumPy image array.

    Supports Poisson noise, additive Gaussian noise, and PSF blur with
    optional noise.

    Args:
        data (np.ndarray): Input image as a NumPy array.
        mode (str, optional): Corruption type. One of "poisson", "gaussian",
            or "psf". Defaults to "poisson".
        size (int, optional): Size parameter for the PSF kernel. Defaults to 64.
        sigma (float, optional): Standard deviation for Gaussian noise.
            Defaults to 0.01.

    Returns:
        np.ndarray: Corrupted image.
    """
    if mode == "poisson":
        data = np.poisson(np.abs(data) * 0.5)
    elif mode == "gaussian":
        data = data + np.random.normal(0, sigma, size=data.shape)
    elif mode == "psf":
        dpsf = psf(size, nrad=4)
        data = scipy.signal.fftconvolve(data, dpsf, "same")
        data = data + np.random.normal(0, sigma, size=data.shape)
    return data


def corrupt(
    data: torch.Tensor, mode: str = "poisson", size=64, sigma=0.01, nrad=5
):
    """Applies synthetic corruption to a torch.Tensor image.

    Supports Poisson noise, additive Gaussian noise, and PSF blur with
    optional noise. Operates on 3D tensors (C, H, W) or batched images.

    Args:
        data (torch.Tensor): Input image tensor.
        mode (str, optional): Corruption type. One of "poisson", "gaussian",
            or "psf". Defaults to "poisson".
        size (int, optional): Size of the PSF kernel for "psf" mode.
            Defaults to 64.
        sigma (float, optional): Standard deviation for Gaussian noise.
            Defaults to 0.01.
        nrad (int, optional): Radial frequency scaling for PSF generation.
            Defaults to 5.

    Returns:
        torch.Tensor: Corrupted image tensor.
    """
    if mode == "poisson":
        data = torch.poisson(torch.abs(data) * 0.5)
    elif mode == "gaussian":
        data = data + torch.normal(0, sigma, size=data.shape)
    elif mode == "psf":
        dpsf = psf(size, nrad=nrad)
        data = scipy.signal.fftconvolve(data[0, :, :].cpu(), dpsf, "same")
        data = torch.tensor(data[None, :, :], dtype=torch.float32)
        data = data + torch.normal(0, sigma, size=data.shape, device="cuda:0")
    return data


def compute_psnr(img1, img2):
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors.

    Converts tensors to NumPy arrays and computes PSNR using the dynamic
    range of the reference image.

    Args:
        img1 (torch.Tensor): Reconstructed or processed image tensor.
        img2 (torch.Tensor): Reference (ground truth) image tensor.

    Returns:
        float: PSNR value in decibels (dB).
    """
    img1_np = img1.squeeze().cpu().numpy()
    img2_np = img2.squeeze().cpu().numpy()
    data_range = img2_np.max() - img2_np.min()
    return psnr(image_true=img2_np, image_test=img1_np, data_range=data_range)


def compute_ssim(img1, img2):
    """Computes the Structural Similarity Index (SSIM) between two tensors.

    Converts input tensors to NumPy arrays and computes SSIM using the
    dynamic range of the reference image.

    Args:
        img1 (torch.Tensor): Reconstructed or processed image tensor.
        img2 (torch.Tensor): Reference (ground truth) image tensor.

    Returns:
        float: SSIM value in the range [0, 1], where higher is better.
    """
    img1_np = img1.squeeze().cpu().numpy()
    img2_np = img2.squeeze().cpu().numpy()
    return ssim(
        img1_np,
        img2_np,
        data_range=img2_np.max() - img2_np.min(),
        channel_axis=0 if img1_np.ndim == 3 else None,
    )
