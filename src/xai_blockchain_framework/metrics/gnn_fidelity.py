"""Node-level fidelity metrics for graph neural networks.

These helpers mirror the tabular metrics in
:mod:`xai_blockchain_framework.metrics.fidelity` but operate on a single
target node of a graph. They match the reference research-code
implementation byte-for-byte:

- Features of the target node are set to zero (comprehensiveness) or
  everything *except* the top-k is set to zero (sufficiency).
- The edge structure is left untouched.
- Infidelity follows the Yeh et al. (2019) definition
  ``(I · attr - (f(x) - f(x - I)))**2`` for Gaussian perturbations of the
  target node's features.

The functions are intentionally side-effect free (they clone the graph
feature matrix on every perturbation) so that caller code can loop over
nodes without worrying about corrupting the shared graph tensor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


# A ``gnn_forward`` callable takes the perturbed feature matrix ``x`` and
# returns the per-node logits. Returning logits keeps the computation graph
# out of the picture (the helpers call ``softmax`` themselves).
GnnForward = Callable[[torch.Tensor], torch.Tensor]


def _fraud_prob(logits: torch.Tensor, node_idx: int) -> float:
    return float(F.softmax(logits[node_idx], dim=0)[1].item())


def gnn_comprehensiveness(
    forward: GnnForward,
    x: torch.Tensor,
    node_idx: int,
    attribution: NDArray[np.float64],
    k_values: list[int] = (1, 3, 5, 10),
) -> dict[int, float]:
    """Comprehensiveness@k for a single node.

    For each ``k`` in ``k_values``, zero out the top-k most important
    features of the target node and measure the drop in fraud probability.

    Returns a dict ``{k: drop}``. ``k`` values larger than the feature
    dimension are skipped.
    """
    with torch.no_grad():
        original = _fraud_prob(forward(x), node_idx)

    ranking = np.argsort(-np.abs(attribution)).copy()
    scores: dict[int, float] = {}
    for k in k_values:
        if k > len(attribution):
            continue
        x_masked = x.clone()
        x_masked[node_idx, ranking[:k]] = 0.0
        with torch.no_grad():
            scores[int(k)] = original - _fraud_prob(forward(x_masked), node_idx)
    return scores


def gnn_sufficiency(
    forward: GnnForward,
    x: torch.Tensor,
    node_idx: int,
    attribution: NDArray[np.float64],
    k_values: list[int] = (1, 3, 5, 10),
) -> dict[int, float]:
    """Sufficiency@k for a single node.

    Keep only the top-k features on the target node and zero out everything
    else. Return ``{k: drop}`` between the original and the kept-only
    prediction.
    """
    with torch.no_grad():
        original = _fraud_prob(forward(x), node_idx)

    ranking = np.argsort(-np.abs(attribution)).copy()
    scores: dict[int, float] = {}
    n_features = x.shape[1]
    for k in k_values:
        if k > len(attribution):
            continue
        x_kept = x.clone()
        drop_mask = torch.ones(n_features, dtype=torch.bool, device=x.device)
        drop_mask[ranking[:k]] = False
        x_kept[node_idx, drop_mask] = 0.0
        with torch.no_grad():
            scores[int(k)] = original - _fraud_prob(forward(x_kept), node_idx)
    return scores


def gnn_infidelity(
    forward: GnnForward,
    x: torch.Tensor,
    node_idx: int,
    attribution: NDArray[np.float64],
    n_perturbations: int = 50,
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """Infidelity for a single node.

    Gaussian perturbations of standard deviation ``sigma`` are applied to
    the target node's feature vector (subtracted, matching the research
    code). The return value is the mean of
    ``(I · attribution - (f(x) - f(x - I)))**2`` over ``n_perturbations``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    device = x.device
    attribution = np.asarray(attribution, dtype=np.float64)
    with torch.no_grad():
        original = _fraud_prob(forward(x), node_idx)

    errors: list[float] = []
    for _ in range(n_perturbations):
        noise = rng.normal(0.0, sigma, size=(x.shape[1],)).astype(np.float32)
        x_pert = x.clone()
        x_pert[node_idx] -= torch.tensor(noise, device=device)
        with torch.no_grad():
            perturbed = _fraud_prob(forward(x_pert), node_idx)
        expl_delta = float(np.dot(attribution, noise))
        model_delta = original - perturbed
        errors.append((expl_delta - model_delta) ** 2)
    return float(np.mean(errors))
