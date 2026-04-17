"""Heatmap helper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap(
    data: pd.DataFrame,
    title: str,
    annot: bool = True,
    fmt: str = ".3f",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    save_to: Path | str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Axes:
    """Render a DataFrame as a Seaborn heatmap with sensible defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax
