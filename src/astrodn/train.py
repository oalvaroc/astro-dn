import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm


def train(
    epochs: int,
    model: nn.Module,
    dataset: data.Dataset,
    optim: nn.Module,
    loss_fn=nn.MSELoss(),
    batch_size=32,
):
    dl = data.DataLoader(dataset, batch_size=batch_size)
    losses = []

    for epoch in range(epochs):
        print(f"> EPOCH {epoch}")
        epoch_loss = 0

        for _, batch in tqdm(enumerate(dl), total=len(dl)):
            optim.zero_grad()

            x, y = batch
            ypred = model(x)
            loss = loss_fn(ypred, y)
            loss.backward()

            epoch_loss += loss.item()

            optim.step()

        epoch_loss /= len(dl)
        losses.append(epoch_loss)

    return losses
