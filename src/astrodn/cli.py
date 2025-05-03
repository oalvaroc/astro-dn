"""AstroDN CLI entry point."""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import random_split
from torchvision.transforms import v2

import astrodn.plot as plot
import astrodn.utils as utils
from astrodn import deconv
from astrodn.dataset import NPYDataset
from astrodn.model import AstroDnNet
from astrodn.train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

MODELS = ("baseline", "v1", "v2")


def create_parser():
    """Creates an argument parser for the CLI tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Tool to train and evaluate astronomical image denoising CNN models."  # noqa: E501
    )

    parser.add_argument("model", type=str, choices=MODELS, help="Model name")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of image patches to feed the model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Directory to save output files: checkpoints, metrics and plots",
    )

    return parser


def get_model(model_name: str):
    """Returns an instance of the specified AstroDnNet model variant.

    Args:
        model_name (str): The name of the model variant. Must be one of
            "baseline", "v1", or "v2".

    Returns:
        AstroDnNet: An instance of the requested model variant.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name == "baseline":
        return AstroDnNet(1)
    elif model_name == "v1":
        return AstroDnNet(1, padding_mode="reflect")
    elif model_name == "v2":
        return AstroDnNet(1, padding_mode="reflect", bn="first")
    else:
        raise ValueError(f"Invalid model {model_name}")


def main():
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%F_%H-%M-%S")
    base_dir = os.path.join(args.output_dir, timestamp)

    try:
        os.makedirs(base_dir)
    except OSError:
        print(f"Directory {base_dir} already exists!")
        sys.exit(1)

    generator = torch.Generator(device=device).manual_seed(1)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.CenterCrop(args.patch_size),
            v2.Lambda(
                lambda x: utils.corrupt(
                    x, mode="psf", size=32, sigma=0.2, nrad=8
                )
            ),
        ]
    )
    raw_transform = v2.Compose([v2.ToImage(), v2.CenterCrop(args.patch_size)])

    ds = NPYDataset(
        "dataset",
        transform=transform,
        target_transform=transform,
        raw_transform=raw_transform,
    )

    train_ds, val_ds, test_ds = random_split(
        ds, [0.50, 0.30, 0.20], generator=generator
    )
    model = get_model(args.model)

    train_loss, val_loss, val_psnr, val_ssim, ref_psnr, ref_ssim = train(
        epochs=args.epochs,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=torch.nn.L1Loss(),
        batch_size=args.batch_size,
        optim=torch.optim.SGD(model.parameters(), lr=0.001),
        ckpt_dir=os.path.join(base_dir, "ckpt"),
    )

    utils.save_losses(
        train_loss, filename=os.path.join(base_dir, "losses-train.csv")
    )
    plot.plot(
        train_loss,
        filename=os.path.join(base_dir, "losses-train.png"),
        dpi=300,
    )
    utils.save_losses(
        val_loss, filename=os.path.join(base_dir, "losses-val.csv")
    )
    plot.plot(
        val_loss, filename=os.path.join(base_dir, "losses-val.png"), dpi=300
    )
    utils.save_losses(val_psnr, filename=os.path.join(base_dir, "psnr.csv"))
    plot.plot(
        {"Validation PSNR": val_psnr, "Baseline": ref_psnr},
        filename=os.path.join(base_dir, "psnr.png"),
        dpi=300,
        title="PSNR",
        ylabel="PSNR",
    )
    utils.save_losses(val_ssim, filename=os.path.join(base_dir, "ssim.csv"))
    plot.plot(
        {"Validation SSIM": val_ssim, "Baseline": ref_ssim},
        filename=os.path.join(base_dir, "ssim.png"),
        dpi=300,
        title="SSIM",
        ylabel="SSIM",
    )
    model.eval()

    with torch.no_grad():
        data, target, ground = test_ds[0]
        data = data.unsqueeze(0)
        pred = model(data)
        plot.plot_image(
            data[0, 0, :, :].cpu().detach().numpy(),
            norm=True,
            filename=os.path.join(base_dir, "patch-input.png"),
            dpi=300,
        )
        plot.plot_image(
            pred[0, 0, :, :].cpu().detach().numpy(),
            norm=True,
            filename=os.path.join(base_dir, "patch-pred.png"),
            dpi=300,
        )
        plot.plot_image(
            target[0, :, :].cpu().detach().numpy(),
            norm=True,
            filename=os.path.join(base_dir, "patch-target.png"),
            dpi=300,
        )
        plot.plot_image(
            ground[0, :, :].cpu().detach().numpy(),
            norm=True,
            filename=os.path.join(base_dir, "patch-groundtruth.png"),
            dpi=300,
        )

    fits_img = os.path.join("sample", "hst_12174_01_wfc3_uvis_f373n_drz.fits")
    data = utils.loadfits(fits_img)
    data = torch.tensor(data[None, None, :, :], dtype=torch.float32)
    plot.plot_image(
        data[0, 0, :, :].cpu().detach().numpy(),
        norm=True,
        filename=os.path.join(base_dir, "input.png"),
        dpi=300,
    )

    out = utils.process_image_with_model(
        model,
        data,
        args.patch_size,
        output_patch_size=args.patch_size,
        stride=args.patch_size // 2,
        batch_size=args.batch_size,
    )

    plot.plot_image(
        out[0, 0, :, :].cpu().detach().numpy(),
        norm=True,
        filename=os.path.join(base_dir, "output.png"),
        dpi=300,
    )

    metrics = {
        "wiener": deconv.evaluate(test_ds, model="wiener", base_dir=base_dir),
        "rl": deconv.evaluate(test_ds, model="rl", base_dir=base_dir),
        "cnn": deconv.evaluate(test_ds, model=model, base_dir=base_dir),
    }

    with open(f"{base_dir}/metrics.json", "w+") as f:
        json.dump(metrics, f, indent=4)
