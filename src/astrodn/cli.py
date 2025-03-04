"""AstroDN CLI entry point."""

import argparse

import torch
from torchvision.transforms import v2

import astrodn.plot as plot
import astrodn.utils as utils
from astrodn.dataset import HSTDataset
from astrodn.model import Baseline
from astrodn.train import train

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

    transform = v2.Compose([v2.ToImage(), v2.GaussianNoise()])
    target_transform = v2.Compose([v2.ToImage(), v2.CenterCrop(248)])

    ds = HSTDataset(
        "dataset",
        patchsize=args.patch_size,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    model = (MODELS.get(args.model))(1)

    losses = train(
        epochs=args.epochs,
        model=model,
        dataset=ds,
        batch_size=args.batch_size,
        optim=torch.optim.Adam(model.parameters()),
    )

    plot.plot_loss(losses, filename="losses.png")
    model.eval()

    data = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        pred = model(data)
        plot.plot_image(
            pred[0, 0, :, :].detach().numpy(), filename="patch.png"
        )

    data = torch.randn(1, 1, 1024, 1024)
    out = utils.process_image_with_model(
        model,
        data,
        args.patch_size,
        output_patch_size=248,
        stride=64,
        batch_size=args.batch_size,
    )

    plot.plot_image(
        data[0, 0, :, :].cpu().detach().numpy(), filename="input.png"
    )
    plot.plot_image(
        out[0, 0, :, :].cpu().detach().numpy(), filename="output.png"
    )
