"""AstroDN CLI entry point."""

import argparse

import numpy as np
import torch
from torchvision.transforms import v2

from astrodn.dataset import HSTDataset
from astrodn.model import Baseline
from astrodn.plot import plot_image

MODELS = {
    "baseline": Baseline,
}


def create_parser():
    """Creates an argument parser for the CLI tool.

    The CLI supports the following commands:
        - train: Train a model with specified parameters.
        - eval: Evaluate a trained model using a given checkpoint.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Tool to train and evaluate astronomical image denoising CNN models."  # noqa: E501
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train", help="Train a CNN model")
    train_parser.add_argument(
        "model", type=str, choices=MODELS.keys(), help="Model name"
    )
    train_parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of image patches to feed the model",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )

    # eval command
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate a trained model"
    )
    eval_parser.add_argument(
        "checkpoint", type=str, help="Path to pytorch model checkpoint file"
    )

    return parser


def main():
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    transform = v2.Compose([
        v2.ToImage()
    ])

    ds = HSTDataset(
        "dataset",
        patchsize=args.patch_size,
        train=True,
        download=True,
        transform=transform,
    )
    model = (MODELS.get(args.model))(1)

    inp, _ = ds[0]
    inp = inp[None, :, :, :]
    out = model(inp)

    print(f"input: {inp.shape}  output: {out.shape}")
    
    with torch.no_grad():
        plot_image(inp[0, 0, :, :].cpu(), filename="input.png")
        plot_image(out[0, 0, :, :].cpu(), filename="output.png")