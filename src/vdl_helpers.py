"""Generic helper utilities for the VDL image captioning mini-challenge.

This module intentionally contains no project-specific model or dataset logic.
It is meant for reproducibility, plotting, lightweight persistence, and device
setup that can be reused from the notebook.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available torch device without assuming CUDA."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_for_apple_silicon() -> None:
    """Apply conservative PyTorch settings for Apple Silicon / MPS workflows."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj: Any, path: str | Path) -> None:
    """Save a JSON-serializable object with a stable, readable format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(
    history: Mapping[str, Sequence[float]],
    metrics: Sequence[str] | None = None,
    title: str = "Training Curves",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot selected metric curves from a history dictionary."""
    if metrics is None:
        metrics = list(history.keys())

    fig, ax = plt.subplots(figsize=figsize)
    for metric in metrics:
        values = history.get(metric)
        if values is None:
            continue
        ax.plot(range(1, len(values) + 1), values, marker="o", label=metric)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_image_with_captions(
    image: Image.Image | np.ndarray | torch.Tensor | str | Path,
    captions: Sequence[str],
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
) -> plt.Figure:
    """Show one image and one or more captions."""
    image_array = _to_display_image(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_array)
    ax.axis("off")

    caption_text = "\n".join(f"{i + 1}. {caption}" for i, caption in enumerate(captions))
    ax.set_title(title or caption_text, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def plot_attention_overlay(
    image: Image.Image | np.ndarray | torch.Tensor | str | Path,
    attention_map: np.ndarray | torch.Tensor,
    title: str | None = None,
    alpha: float = 0.45,
    cmap: str = "magma",
    figsize: tuple[int, int] = (6, 6),
) -> plt.Figure:
    """Overlay a spatial attention map on an image."""
    image_array = _to_display_image(image)
    attention = _to_numpy(attention_map).astype(float)

    if attention.ndim != 2:
        raise ValueError("attention_map must be a 2D array")

    attention = attention - attention.min()
    max_value = attention.max()
    if max_value > 0:
        attention = attention / max_value

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_array)
    ax.imshow(attention, cmap=cmap, alpha=alpha, extent=ax.images[0].get_extent())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensors or arrays to a detached NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_display_image(image: Image.Image | np.ndarray | torch.Tensor | str | Path) -> np.ndarray:
    """Convert common image inputs to an HWC NumPy array for matplotlib."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        return np.asarray(image)

    array = _to_numpy(image)

    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)

    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0)

    return array

