from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ("cats", "dogs")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = root / class_name
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((path, label) for path in files)
    return samples


def stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[Path, int]]] = {index: [] for index in range(len(CLASS_NAMES))}
    for sample in samples:
        grouped[sample[1]].append(sample)

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    for label_samples in grouped.values():
        label_samples = label_samples[:]
        rng.shuffle(label_samples)
        split_index = int(len(label_samples) * (1 - val_ratio))
        train_samples.extend(label_samples[:split_index])
        val_samples.extend(label_samples[split_index:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


class ImageClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def count_by_label(samples: list[tuple[Path, int]]) -> list[int]:
    counts = [0] * len(CLASS_NAMES)
    for _, label in samples:
        counts[label] += 1
    return counts


def class_weights_from_samples(samples: list[tuple[Path, int]]) -> torch.Tensor:
    counts = count_by_label(samples)
    max_count = max(counts)
    weights = [max_count / count for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: tuple[str, ...]
    class_weights: torch.Tensor
    train_counts: list[int]
    val_counts: list[int]
    test_counts: list[int]


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = 128,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    data_dir = Path(data_dir)
    train_samples_full = scan_samples(data_dir / "training_set")
    test_samples = scan_samples(data_dir / "test_set")
    train_samples, val_samples = stratified_split(train_samples_full, val_ratio=val_ratio, seed=seed)
    train_transform, eval_transform = build_transforms(image_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        ImageClassificationDataset(train_samples, train_transform),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ImageClassificationDataset(val_samples, eval_transform),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        ImageClassificationDataset(test_samples, eval_transform),
        shuffle=False,
        **loader_kwargs,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        class_weights=class_weights_from_samples(train_samples),
        train_counts=count_by_label(train_samples),
        val_counts=count_by_label(val_samples),
        test_counts=count_by_label(test_samples),
    )
