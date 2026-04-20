"""GNN explainers: GNNExplainer, Integrated Gradients, and GraphLIME.

This module is a direct port of the research-code notebook
``03b_xai_gnn_explainer.ipynb``. The logic is preserved byte-for-byte so
that the published results can be reproduced.

Three entry points are provided:

- :func:`run_gnnexplainer` uses PyTorch Geometric's GNNExplainer with 200
  epochs and learning rate 0.01, on feature-level node masks.
- :func:`run_ig` wraps the model in a minimal ``ModelWrapper`` that holds
  ``edge_index`` as a plain Python attribute (not a buffer) and drops any
  ``ts`` argument, then applies Captum's :class:`IntegratedGradients` with
  50 integration steps and a zero baseline. The absolute value of the
  attribution at the target node is returned.
- :func:`run_graphlime` is a simple feature-perturbation GraphLIME: for
  each feature of the target node, replace its value by the value at a
  random donor node ten times and average the absolute change in fraud
  probability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from numpy.typing import NDArray
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


# ---------------------------------------------------------------------------
# GNNExplainer
# ---------------------------------------------------------------------------

def run_gnnexplainer(
    model: nn.Module,
    data,
    node_indices,
    name: str = "",
    epochs: int = 200,
    lr: float = 1e-2,
    verbose_every: int = 25,
) -> NDArray[np.float32]:
    """Feature-level node-mask explanations via PyG's GNNExplainer.

    Parameters
    ----------
    model : nn.Module
        Trained GNN (TemporalGCN or GraphSAGE).
    data : torch_geometric.data.Data
        Graph data with ``x``, ``edge_index``, ``num_features``.
    node_indices : torch.Tensor
        Indices of the nodes to explain.
    name : str, optional
        Printed with progress lines.
    epochs, lr : int, float
        Hyperparameters of the GNNExplainer optimizer.
    verbose_every : int
        Print progress every ``verbose_every`` nodes.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(node_indices), num_features)``, dtype float32.
    """
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    attrs: list[NDArray[np.float32]] = []
    for i, nid in enumerate(node_indices):
        try:
            exp = explainer(data.x, data.edge_index, index=nid.item())
            mask = exp.node_mask[nid].detach().cpu().numpy()
            attrs.append(mask)
        except Exception:
            attrs.append(np.zeros(data.num_features))
        if (i + 1) % verbose_every == 0:
            print(f"  {name}: {i + 1}/{len(node_indices)}")
    return np.array(attrs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    """Port of the research-code ``ModelWrapper`` used for Captum's IG.

    Notes
    -----
    Two details are load-bearing and must not be changed:

    1. ``edge_index`` is stored as a plain Python attribute (not a
       :meth:`torch.nn.Module.register_buffer` call). Captum walks the
       module's buffers when tracking gradient flow, and registering
       ``edge_index`` as a buffer has been observed to trigger massive
       memory allocation during IG on GraphSAGE, slowing the notebook by
       orders of magnitude. The original research code uses a plain
       attribute, and so do we.
    2. The wrapper's ``forward`` calls ``self.model(x, self.edge_index)``
       without any ``ts`` argument, even when the underlying model is a
       TemporalGCN. The TemporalGCN already handles ``ts=None`` by
       substituting a zero time encoding, and this behavior is preserved
       here for fidelity with the original run.
    """

    def __init__(self, model: nn.Module, edge_index: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.edge_index = edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.edge_index)


def run_ig(
    model: nn.Module,
    data,
    node_indices,
    name: str = "",
    target_class: int = 1,
    n_steps: int = 50,
    verbose_every: int = 25,
) -> NDArray[np.float32]:
    """Captum Integrated Gradients attributions for a set of nodes.

    Parameters
    ----------
    model : nn.Module
        Trained GNN.
    data : torch_geometric.data.Data
        Graph data with ``x``, ``edge_index``, ``num_features``.
    node_indices : torch.Tensor
        Indices of the nodes to explain.
    target_class : int, default 1
        Logit index to attribute.
    n_steps : int, default 50
        Number of integration steps.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(node_indices), num_features)``, dtype float32,
        with the absolute IG values at the target node.
    """
    model.eval()
    wrapped = ModelWrapper(model, data.edge_index)
    ig = IntegratedGradients(wrapped)
    baseline = torch.zeros_like(data.x)
    attrs: list[NDArray[np.float32]] = []
    for i, nid in enumerate(node_indices):
        try:
            attr, _ = ig.attribute(
                data.x,
                baselines=baseline,
                target=target_class,
                return_convergence_delta=True,
                n_steps=n_steps,
            )
            attrs.append(np.abs(attr[nid].detach().cpu().numpy()))
        except Exception:
            attrs.append(np.zeros(data.num_features))
        if (i + 1) % verbose_every == 0:
            print(f"  {name}: {i + 1}/{len(node_indices)}")
    return np.array(attrs, dtype=np.float32)


# ---------------------------------------------------------------------------
# GraphLIME (simple feature-perturbation variant)
# ---------------------------------------------------------------------------

class SimpleGraphLIME:
    """Port of the research-code feature-perturbation GraphLIME explainer.

    For each feature of the target node, replace its value by the value
    at a random donor node ``n_pert`` times and record the mean absolute
    change in fraud probability. The resulting importance vector is the
    attribution for that node.
    """

    def __init__(self, model: nn.Module, data, n_pert: int = 10) -> None:
        self.model = model
        self.data = data
        self.n_pert = n_pert

    def explain(self, node_idx) -> NDArray[np.float64]:
        self.model.eval()
        nid = node_idx if isinstance(node_idx, int) else node_idx.item()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            orig = F.softmax(out[nid], dim=0)[1].item()
        nf = self.data.x.shape[1]
        imp = np.zeros(nf)
        for fi in range(nf):
            preds = []
            for _ in range(self.n_pert):
                xp = self.data.x.clone()
                xp[nid, fi] = self.data.x[np.random.randint(0, self.data.num_nodes), fi]
                with torch.no_grad():
                    p = F.softmax(self.model(xp, self.data.edge_index)[nid], dim=0)[1].item()
                preds.append(p)
            imp[fi] = abs(orig - np.mean(preds))
        return imp


def make_gnn_ig_explain_fn(
    model: nn.Module,
    graph_data,
    node_idx: int,
    ts_tensor: torch.Tensor | None = None,
    n_steps: int = 20,
    target_class: int = 1,
) -> callable:
    """Return a single-instance IG explanation function for stability metrics.

    The returned callable accepts a 1-D numpy array (the feature vector of
    ``node_idx``), substitutes it into the shared graph, runs Captum
    :class:`IntegratedGradients` on the resulting node logit, and returns
    the IG attribution for that node as a 1-D numpy array.

    This helper exists because the Module 2 (stability) bootstrap procedure
    calls the explainer repeatedly on perturbed copies of a single node's
    features, and needs an ``explain(x) -> attr`` contract. The target-node
    index is captured in the closure.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN.
    graph_data : torch_geometric.data.Data
        Graph object; ``graph_data.x`` and ``graph_data.edge_index`` are
        read on every call so the underlying tensors must remain valid.
    node_idx : int
        Node whose feature vector will be perturbed.
    ts_tensor : torch.Tensor, optional
        Time tensor (only required for TemporalGCN); automatically ignored
        when ``model`` has no ``time_enc`` layer.
    n_steps : int, default 20
        Number of IG integration steps.
    target_class : int, default 1
        Logit index to attribute.
    """
    has_ts = hasattr(model, "time_enc")
    device = graph_data.x.device

    def model_forward(x_node: torch.Tensor) -> torch.Tensor:
        x_full = graph_data.x.clone()
        x_full[node_idx] = x_node[0] if x_node.dim() > 1 else x_node
        if has_ts:
            out = model(x_full, graph_data.edge_index, ts=ts_tensor)
        else:
            out = model(x_full, graph_data.edge_index)
        return out[node_idx: node_idx + 1]

    ig = IntegratedGradients(model_forward)

    def explain_fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_flat = np.asarray(x, dtype=np.float32).flatten()
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device).unsqueeze(0)
        x_tensor.requires_grad = True
        attr = ig.attribute(x_tensor, target=target_class, n_steps=n_steps)
        return attr.squeeze().detach().cpu().numpy()

    return explain_fn


def run_graphlime(
    model: nn.Module,
    data,
    node_indices,
    name: str = "",
    n_pert: int = 10,
    verbose_every: int = 25,
) -> NDArray[np.float32]:
    """Run :class:`SimpleGraphLIME` on a batch of nodes."""
    gl = SimpleGraphLIME(model, data, n_pert=n_pert)
    attrs: list[NDArray[np.float32]] = []
    for i, nid in enumerate(node_indices):
        try:
            attrs.append(gl.explain(nid))
        except Exception:
            attrs.append(np.zeros(data.num_features))
        if (i + 1) % verbose_every == 0:
            print(f"  {name}: {i + 1}/{len(node_indices)}")
    return np.array(attrs, dtype=np.float32)
