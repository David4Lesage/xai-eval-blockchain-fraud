"""Bar chart helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def bar_chart(
    labels: Sequence[str],
    values: ArrayLike,
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    save_to: Path | str | None = None,
    figsize: tuple[int, int] = (14, 7),
    value_format: str = "{:.3f}",
    ylim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Draw a single-series bar chart with per-bar annotations.

    Parameters
    ----------
    labels : sequence of str
        Bar labels along the x-axis.
    values : array-like
        Bar heights.
    title, ylabel : str
        Chart annotations.
    colors : sequence of str, optional
        Bar colors. If None, matplotlib's default palette is used.
    save_to : path, optional
        Output file. The figure is saved when provided.
    figsize : tuple, default (14, 7)
        Figure dimensions in inches.
    value_format : str, default "{:.3f}"
        Format specifier for the value annotation above each bar.
    ylim : tuple of float, optional
        Explicit y-axis range.

    Returns
    -------
    matplotlib.axes.Axes
    """
    values = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)
    for bar, val in zip(bars, values):
        ax.annotate(
            value_format.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax


def log_bar_chart(
    labels: Sequence[str],
    values: ArrayLike,
    title: str,
    ylabel: str,
    **kwargs,
) -> plt.Axes:
    """Bar chart with a ``log10(1 + x)`` transform on the values.

    Useful for metrics that span multiple orders of magnitude (Lipschitz,
    CoV). The annotation shows the raw value in parentheses.
    """
    raw = np.asarray(values, dtype=np.float64)
    logged = np.log10(1.0 + raw)
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (14, 7)))
    colors = kwargs.pop("colors", None)
    bars = ax.bar(
        range(len(logged)), logged, color=colors,
        alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val_log, val_raw in zip(bars, logged, raw):
        ax.annotate(
            f"{val_log:.2f}\n(raw: {val_raw:.1f})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=6,
        )
    fig.tight_layout()
    save_to = kwargs.pop("save_to", None)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax


def grouped_bar_chart(
    labels: Sequence[str],
    series: dict[str, ArrayLike],
    title: str,
    ylabel: str,
    save_to: Path | str | None = None,
    figsize: tuple[int, int] = (14, 7),
) -> plt.Axes:
    """Draw a grouped bar chart from a dict of ``name -> values``."""
    fig, ax = plt.subplots(figsize=figsize)
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups)
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, np.asarray(values), width, label=name, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax
