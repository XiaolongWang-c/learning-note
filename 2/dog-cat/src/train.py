from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils import (
    accuracy_from_logits,
    ensure_dir,
    plot_histories,
    save_checkpoint,
    save_history_csv,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cat-dog classifiers.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--model", choices=["plain", "res", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += accuracy_from_logits(logits, labels) * batch_size
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_one_model(model_name: str, args: argparse.Namespace, bundle, device: torch.device) -> dict[str, list[float]]:
    model_dir = ensure_dir(args.output_dir / model_name)
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=bundle.class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, bundle.train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, bundle.val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[{model_name}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_name": model_name,
            "val_acc": val_acc,
            "args": vars(args),
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model_dir / "best.pt", model, payload)

    save_checkpoint(
        model_dir / "last.pt",
        model,
        {"epoch": args.epochs, "model_name": model_name, "val_acc": history["val_acc"][-1], "args": vars(args)},
    )
    save_history_csv(history, model_dir / "history.csv")
    save_json(
        {
            "model": model_name,
            "best_val_acc": best_val_acc,
            "train_counts": bundle.train_counts,
            "val_counts": bundle.val_counts,
            "test_counts": bundle.test_counts,
        },
        model_dir / "summary.json",
    )
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    output_dir = ensure_dir(args.output_dir)
    model_names = ["plain", "res"] if args.model == "both" else [args.model]
    histories: dict[str, dict[str, list[float]]] = {}
    for model_name in model_names:
        histories[model_name] = train_one_model(model_name, args, bundle, device)

    plot_histories(histories, output_dir / "figures")
    save_json(
        {
            "device": str(device),
            "class_names": list(bundle.class_names),
            "train_counts": bundle.train_counts,
            "val_counts": bundle.val_counts,
            "test_counts": bundle.test_counts,
            "models": model_names,
        },
        output_dir / "run_config.json",
    )


if __name__ == "__main__":
    main()
