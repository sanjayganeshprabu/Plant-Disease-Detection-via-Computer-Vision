"""Figures used by the training / evaluation / prediction scripts.

Uses ``matplotlib.figure.Figure`` directly (not pyplot) so it works on
servers and in Docker without a display; every function saves a PNG.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def plot_training_history(history: dict, out_path: Path) -> Path:
    fig = Figure(figsize=(14, 5))
    ax_loss, ax_acc = fig.subplots(1, 2)
    epochs = range(1, len(history["loss"]) + 1)

    ax_loss.plot(epochs, history["loss"], marker="o", label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    ax_loss.set(title="Train and Validation Loss", xlabel="Epoch", ylabel="Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["accuracy"], marker="o", label="Train accuracy")
    ax_acc.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    ax_acc.set(title="Train and Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    return out_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Sequence[str], out_path: Path) -> Path:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig = Figure(figsize=(24, 22))
    ax = fig.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation="vertical", cmap="Blues", colorbar=True)
    for text in disp.text_.ravel():  # 38x38 grid -> keep numbers small
        text.set_fontsize(7)
    ax.set_title("Confusion Matrix", fontsize=18)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    return out_path


def plot_prediction_grid(image_paths: Sequence[Path], predictions: Sequence, out_path: Path,
                         cols: int = 3, max_images: int = 9) -> Path:
    """Grid of images with predicted labels; unhealthy leaves get a red tint
    (same idea as the last cell of the original notebook)."""
    items = list(zip(image_paths, predictions))[:max_images]
    rows = max(1, math.ceil(len(items) / cols))
    fig = Figure(figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (path, pred) in zip(axes, items):
        img = np.asarray(Image.open(path).convert("RGB")).astype("float32")
        if not pred.is_healthy:
            red = np.zeros_like(img)
            red[..., 0] = 255
            img = 0.7 * img + 0.3 * red
            label, color = f"{pred.plant}: {pred.condition}", "crimson"
        else:
            label, color = f"{pred.plant}: Healthy", "green"
        ax.imshow(img.astype("uint8"))
        ax.set_title(f"{Path(path).name}\n{label} ({pred.confidence:.0%})", color=color, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    return out_path
