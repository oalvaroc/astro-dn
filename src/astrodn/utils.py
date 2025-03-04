import astropy.io.fits as fits
import numpy as np
import torch
import torch.nn as nn


def loadfits(file):
    """Load FITS image file."""
    return fits.getdata(file).astype(np.float32)


def split_into_patches(
    image: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """Splits an image into overlapping patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        patch_size (int): The size of the extracted patches.
        stride (int): The stride between patches when splitting.

    Returns:
        torch.Tensor: Extracted patches of shape
                      (batch, num_patches, channels, patch_size, patch_size).
    """
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
