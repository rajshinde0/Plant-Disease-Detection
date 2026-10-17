"""Datasets, transforms and device plumbing.

Transforms live HERE and are imported by training, evaluation, the CLI and the
Flask app. That is deliberate: the single most common failure in projects like
this one is inference preprocessing drifting away from training preprocessing,
which silently wrecks accuracy. One definition, one place.

Normalisation is ToTensor() only, mapping pixels to [0, 1], exactly as the
paper describes - no mean/std standardisation.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# device helpers
# --------------------------------------------------------------------------
def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wraps a DataLoader to move batches to a device as they are yielded."""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


def tune_threads(device: torch.device, num_workers: int) -> None:
    """Avoid CPU oversubscription.

    On a 4-core/8-thread laptop, DataLoader workers and torch's intra-op
    threads fight each other and can cost 30%+ of throughput. Leave the
    workers some headroom instead of letting both default to all cores.
    """
    if device.type != "cpu":
        return
    total = os.cpu_count() or 4
    torch.set_num_threads(max(1, total - max(1, num_workers)))


def default_workers(device: torch.device) -> int:
    return 4 if device.type == "cuda" else 2


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def build_transforms(img_size: int = config.IMG_SIZE, train: bool = False,
                     augment: bool = True) -> transforms.Compose:
    """Build the preprocessing pipeline.

    Validation/inference: resize to img_size, convert to a [0, 1] tensor.
    Training (augment=True): the paper's augmentation set - rotation, flipping,
    cropping and brightness variation - kept mild because the Kaggle dataset is
    already offline-augmented from the original PlantVillage images.
    """
    if train and augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# --------------------------------------------------------------------------
# class names
# --------------------------------------------------------------------------
def save_class_names(names, path: Path = config.CLASS_NAMES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(list(names), fh, indent=2)


def load_class_names(path: Path = config.CLASS_NAMES_FILE):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def discover_class_names(root: Path = config.TRAIN_DIR):
    """Read class names straight off disk, so they always match the dataset."""
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


# --------------------------------------------------------------------------
# datasets / loaders
# --------------------------------------------------------------------------
def stratified_subset(dataset: ImageFolder, fraction: float, seed: int = config.SEED):
    """Take the same fraction of every class, preserving balance."""
    if fraction >= 1.0:
        return dataset
    by_class = {}
    for idx, (_path, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    keep = []
    for label in sorted(by_class):
        idxs = by_class[label]
        rng.shuffle(idxs)
        n = max(1, int(round(len(idxs) * fraction)))
        keep.extend(idxs[:n])
    keep.sort()
    return Subset(dataset, keep)


def should_drop_last(n: int, batch_size: int) -> bool:
    """Whether the training loader should drop its final batch.

    A final batch of exactly ONE sample crashes BatchNorm in training mode
    ("Expected more than 1 value per channel"), because at 64px the deepest
    feature map is already 1x1. Dropping that batch costs one image per epoch
    and only happens when the remainder is exactly 1, so no data is wasted in
    the normal case.
    """
    return n > batch_size and n % batch_size == 1


def build_loaders(img_size: int = config.IMG_SIZE,
                  batch_size: int = config.BATCH_SIZE,
                  subset: float = 1.0,
                  augment: bool = True,
                  num_workers: int | None = None,
                  device: torch.device | None = None,
                  train_dir: Path = config.TRAIN_DIR,
                  valid_dir: Path = config.VALID_DIR):
    """Return (train_dl, valid_dl, class_names) ready on the target device."""
    for d in (train_dir, valid_dir):
        if not Path(d).is_dir():
            raise FileNotFoundError(
                "dataset directory missing: %s\nRun: python scripts/prepare_dataset.py" % d)

    device = device or get_default_device()
    if num_workers is None:
        num_workers = default_workers(device)
    tune_threads(device, num_workers)

    train_ds = ImageFolder(str(train_dir), build_transforms(img_size, train=True,
                                                            augment=augment))
    valid_ds = ImageFolder(str(valid_dir), build_transforms(img_size, train=False))
    class_names = train_ds.classes

    if subset < 1.0:
        train_ds = stratified_subset(train_ds, subset)
        valid_ds = stratified_subset(valid_ds, subset)

    pin = device.type == "cuda"
    common = dict(num_workers=num_workers, pin_memory=pin)

    # Only the training loader keeps its workers alive between epochs. Marking
    # the validation loader persistent too would leave a second set of worker
    # processes resident through every training epoch, competing for the same
    # few cores on a laptop.
    train_common = dict(common)
    if num_workers > 0:
        train_common["persistent_workers"] = True

    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True,
        drop_last=should_drop_last(len(train_ds), batch_size), **train_common)
    valid_dl = DataLoader(valid_ds, batch_size * 2, shuffle=False, **common)

    return (DeviceDataLoader(train_dl, device),
            DeviceDataLoader(valid_dl, device),
            class_names)


def build_valid_loader(img_size: int = config.IMG_SIZE,
                       batch_size: int = config.BATCH_SIZE,
                       num_workers: int | None = None,
                       device: torch.device | None = None,
                       valid_dir: Path = config.VALID_DIR):
    """Validation-only loader, for evaluation runs. Also returns the dataset."""
    if not Path(valid_dir).is_dir():
        raise FileNotFoundError(
            "dataset directory missing: %s\nRun: python scripts/prepare_dataset.py" % valid_dir)
    device = device or get_default_device()
    if num_workers is None:
        num_workers = default_workers(device)
    tune_threads(device, num_workers)

    ds = ImageFolder(str(valid_dir), build_transforms(img_size, train=False))
    dl = DataLoader(ds, batch_size * 2, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    return DeviceDataLoader(dl, device), ds
