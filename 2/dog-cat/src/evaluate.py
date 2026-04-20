from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils import (
    confusion_matrix,
    ensure_dir,
    metrics_from_confusion,
    plot_confusion_matrix,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cat-dog classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["plain", "res"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_targets = []

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    matrix = confusion_matrix(preds, targets, num_classes=2)
    return total_loss / total_count, total_correct / total_count, matrix


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    loader = bundle.val_loader if args.split == "val" else bundle.test_loader

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, acc, matrix = evaluate(model, loader, device)
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = loss
    metrics["accuracy"] = acc
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["confusion_matrix"] = matrix.tolist()

    output_dir = ensure_dir(args.checkpoint.parent / "evaluation")
    save_json(metrics, output_dir / f"{args.split}_metrics.json")
    plot_confusion_matrix(matrix, bundle.class_names, output_dir / f"{args.split}_confusion.png")


if __name__ == "__main__":
    main()
