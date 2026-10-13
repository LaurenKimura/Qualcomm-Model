"""Load the satellite photos and the official train/validation split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Same preparation the starter notebook uses. The photos are stretched to
# 224 by 224, then rescaled with the usual ImageNet averages so the network
# sees numbers in a familiar range.
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_eurosat(root: Path, download: bool):
    return datasets.EuroSAT(root=str(root), download=download, transform=EVAL_TRANSFORM)


def make_loaders(
    dataset,
    split_path: Path,
    batch_size: int,
    seed: int = 42,
):
    """Return train and validation loaders.

    Judges score the indices in val_split.json. If that file is missing we
    still build a fixed 80/20 split so the code can be tested, and we say so.
    """
    official = split_path.is_file()
    if official:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = list(split["train"])
        val_idx = list(split["val"])
    else:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        cut = int(0.8 * len(dataset))
        train_idx, val_idx = order[:cut], order[cut:]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, official
