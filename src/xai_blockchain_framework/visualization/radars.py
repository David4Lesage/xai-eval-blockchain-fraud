"""Radar chart helpers for multi-metric comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def radar_chart(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    title: str,
    colors: dict[str, str] | None = None,
    save_to: Path | str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Axes:
    """Draw a radar chart from a dict of ``name -> values``.

    All series must have the same length as ``categories``. Values are
    expected to be already normalized to ``[0, 1]`` where higher is better.

    Parameters
    ----------
    categories : sequence of str
        Axis labels, one per dimension.
    series : dict of str to sequence of float
        Named series, each a list of values aligned with ``categories``.
    title : str
        Chart title.
    colors : dict of str to str, optional
        Mapping from series name to color. Matplotlib defaults otherwise.
    save_to : path, optional
        Output file.
    figsize : tuple, default (10, 8)

    Returns
    -------
    matplotlib.axes.Axes
    """
    n = len(categories)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    for name, values in series.items():
        values = list(values) + [list(values)[0]]
        color = (colors or {}).get(name)
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color, alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax
