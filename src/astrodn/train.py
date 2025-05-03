"""Training utilities for denoising models.

Provides functions to train image denoising models and manage model
checkpoints. Tracks training/validation loss, PSNR, and SSIM metrics across
epochs.
"""

import os

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from . import utils


def checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    nkeep=5,
):
    """Saves a model checkpoint and maintains only the most recent ones.

    Saves the current model and optimizer state along with the epoch number
    to a checkpoint file. If the number of saved checkpoints reaches `nkeep`,
    the oldest one is removed.

    Args:
        ckpt_dir (str): Directory to store checkpoint files.
        model (nn.Module): PyTorch model to checkpoint.
        optim (torch.optim.Optimizer): Optimizer whose state will be saved.
        epoch (int): Current training epoch.
        nkeep (int, optional): Maximum number of checkpoints to keep.
            Defaults to 5.
    """
    files = sorted(os.listdir(ckpt_dir))
    if len(files) == nkeep:
        os.remove(os.path.join(ckpt_dir, files[0]))

    path = os.path.join(ckpt_dir, f"epoch-{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        },
        path,
    )


def train(
    epochs: int,
    model: nn.Module,
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    optim: torch.optim.Optimizer,
    loss_fn=nn.MSELoss(),
    batch_size=32,
    ckpt_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Trains a model and evaluates on a validation set each epoch.

    Tracks loss and denoising performance metrics (PSNR, SSIM) during
    training. Optionally saves checkpoints after each epoch.

    Args:
        epochs (int): Number of training epochs.
        model (nn.Module): Model to train.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        optim (Optimizer): Optimizer for training.
        loss_fn (callable, optional): Loss function to minimize. Defaults to
            nn.MSELoss().
        batch_size (int, optional): Training batch size. Defaults to 32.
        ckpt_dir (str, optional): Directory to save checkpoints. If None,
            no checkpoints are saved. Defaults to None.
        device (str, optional): Device to use ("cuda" or "cpu"). Defaults to
            CUDA if available.

    Returns:
        tuple: Six lists containing epoch-wise values for:
            - training loss
            - validation loss
            - validation PSNR
            - validation SSIM
            - reference PSNR (noisy input)
            - reference SSIM (noisy input)
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dl = data.DataLoader(train_dataset, batch_size=batch_size)
    val_dl = data.DataLoader(val_dataset, batch_size=1)

    model.to(device)
    train_losses, val_losses, val_psnrs, val_ssims, ref_psnrs, ref_ssims = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for epoch in tqdm(range(epochs), unit="epoch"):
        model.train()
        epoch_loss = 0

        for _, batch in tqdm(
            enumerate(train_dl), total=len(train_dl), leave=False
        ):
            optim.zero_grad()
            x, y, _ = [b.to(device) for b in batch]
            ypred = model(x)
            loss = loss_fn(ypred, y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        vloss, vpsnr, vssim = 0, 0, 0
        refpsnr, refssim = 0, 0
        with torch.no_grad():
            for x, y, z in val_dl:
                x, y, z = x.to(device), y.to(device), z.to(device)
                ypred = model(x)
                vloss += loss_fn(ypred, y).item()
                vpsnr += utils.compute_psnr(ypred, z)
                vssim += utils.compute_ssim(ypred, z)
                refpsnr += utils.compute_psnr(x, z)
                refssim += utils.compute_ssim(x, z)

        n_val = len(val_dl)
        val_losses.append(vloss / n_val)
        val_psnrs.append(vpsnr / n_val)
        val_ssims.append(vssim / n_val)
        ref_psnrs.append(refpsnr / n_val)
        ref_ssims.append(refssim / n_val)

        if ckpt_dir:
            checkpoint(ckpt_dir, model, optim, epoch)

    return train_losses, val_losses, val_psnrs, val_ssims, ref_psnrs, ref_ssims
