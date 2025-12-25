from torch.utils.data import TensorDataset, DataLoader
from typing import Callable
import torch

def get_dataloader(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    N: int,
    input_dim: int = 2,
    batch_size: int = 16,
    device: str = "cpu",
    shffle: bool = True
):
    x = torch.empty(N, input_dim).uniform_(-1, 1)
    y = target_fn(x)
    x = x.to(device)
    y = y.to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataset, dataloader

