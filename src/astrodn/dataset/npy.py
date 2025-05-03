"""Dataset loader for .npy image files.

Provides a PyTorch dataset class for loading and transforming .npy files
used in image denoising tasks. Returns noisy inputs, targets, and raw
images for evaluation.
"""

import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset


class NPYDataset(VisionDataset):
    """Dataset for loading .npy image files for denoising tasks."""

    def __init__(
        self, root, transform=None, target_transform=None, raw_transform=None
    ):
        """Initializes the dataset from a directory of .npy files.

        Args:
            root (str): Directory containing .npy files.
            transform (callable, optional): A function/transform that takes
                an input image and returns a transformed version.
            target_transform (callable, optional): A function/transform for
                the target image. Defaults to None.
            raw_transform (callable, optional): A function/transform for the
                raw image used in evaluation (e.g., normalization).
        """
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.raw_transform = raw_transform
        self.input_dir = root

        self.input_files = sorted(
            [
                os.path.join(self.input_dir, f)
                for f in os.listdir(self.input_dir)
                if f.endswith(".npy")
            ]
        )

    def __getitem__(self, index):
        """Loads and returns the noisy, target, and raw image tensors.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple (input, target, raw), where each is a torch.Tensor.
        """
        orig = np.load(self.input_files[index])
        orig = orig.astype(np.float32, copy=False)

        x = torch.from_numpy(orig)
        raw = torch.from_numpy(orig)
        if self.raw_transform:
            raw = self.raw_transform(raw)

        if self.transforms:
            return *self.transforms(x, x), raw
        return x, x, raw

    def __len__(self):
        """Returns the total number of .npy files.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.input_files)
