from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_history_csv(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_histories(histories: dict[str, dict[str, list[float]]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    plots = [
        ("loss_compare.png", "Loss", ("train_loss", "val_loss")),
        ("accuracy_compare.png", "Accuracy", ("train_acc", "val_acc")),
    ]
    for file_name, ylabel, keys in plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, key in zip(axes, keys):
            for model_name, history in histories.items():
                axis.plot(history[key], label=model_name)
            axis.set_title(key)
            axis.set_xlabel("epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=200)
        plt.close(fig)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, pred in zip(targets, preds):
        matrix[target.long(), pred.long()] += 1
    return matrix


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.float()
    correct = matrix.diag().sum().item()
    total = matrix.sum().item()
    precision = []
    recall = []
    f1 = []
    for index in range(matrix.size(0)):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
    return {
        "accuracy": correct / total if total else 0.0,
        "precision_macro": sum(precision) / len(precision),
        "recall_macro": sum(recall) / len(recall),
        "f1_macro": sum(f1) / len(f1),
    }


def plot_confusion_matrix(matrix: torch.Tensor, class_names: tuple[str, ...], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for row in range(matrix.size(0)):
        for col in range(matrix.size(1)):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_checkpoint(path: str | Path, model: torch.nn.Module, payload: dict) -> None:
    state = {"model_state_dict": model.state_dict(), **payload}
    torch.save(state, path)
