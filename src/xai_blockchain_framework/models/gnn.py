"""Graph Neural Network baselines for the Elliptic dataset.

Direct port of the research-code notebook ``02b_Elliptic_Baselines_GNN``.
Two architectures are provided:

- :class:`TemporalGCN` uses two TAGConv layers (K=3 then K=2) followed
  by a continuous time encoding concatenated to the final node
  embedding, with a two-way logit head.
- :class:`GraphSAGEModel` uses two mean-aggregation SAGE layers with a
  two-way logit head.

Training is performed by :func:`train_gnn`, which implements AUC-based
early stopping on the validation mask, capped class weights, gradient
clipping, and a 200-epoch upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv, TAGConv

from xai_blockchain_framework.config import CONFIG


def get_device() -> torch.device:
    """Return the torch device honoring ``CONFIG.torch_device`` if set."""
    preference = CONFIG.torch_device.lower() if CONFIG.torch_device else ""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalGCN(nn.Module):
    """Temporal-GCN with TAGConv layers and an additive time encoding."""

    def __init__(
        self,
        in_c: int,
        hid: int = 128,
        out: int = 2,
        K: int = 3,
        drop: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = TAGConv(in_c, hid, K=K)
        self.conv2 = TAGConv(hid, hid, K=2)
        self.time_enc = nn.Linear(1, hid)
        self.fc = nn.Linear(hid * 2, out)
        self.drop = drop
        self.hid = hid

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ts: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.drop, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.drop, self.training)
        if ts is not None:
            t = (ts.unsqueeze(-1) - ts.min()) / (ts.max() - ts.min() + 1e-8)
            h = torch.cat([h, self.time_enc(t)], -1)
        else:
            h = torch.cat([h, torch.zeros(h.size(0), self.hid, device=h.device)], -1)
        return self.fc(h)


class GraphSAGEModel(nn.Module):
    """Two-layer GraphSAGE with mean aggregation."""

    def __init__(
        self,
        in_c: int,
        hid: int = 128,
        out: int = 2,
        drop: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.fc = nn.Linear(hid, out)
        self.drop = drop

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.drop, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.drop, self.training)
        return self.fc(h)


# Back-compat alias: original notebook uses ``GraphSAGEModel``. The
# newer framework code sometimes imports ``GraphSAGE``. Both point to
# the same class so that both notebooks and the library can use either.
GraphSAGE = GraphSAGEModel


def train_gnn(
    model: nn.Module,
    data,
    epochs: int = 200,
    lr: float = 1e-2,
    patience: int = 30,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> nn.Module:
    """Train a GNN with AUC-based early stopping.

    Parameters
    ----------
    model : nn.Module
        The GNN instance to train (TemporalGCN or GraphSAGE).
    data : torch_geometric.data.Data
        Graph data with ``x``, ``edge_index``, ``y``, ``train_mask``,
        ``val_mask`` and optionally ``ts`` (required for TemporalGCN).
    epochs : int, default 200
    lr : float, default 1e-2
    patience : int, default 30
        Number of epochs without AUC improvement before early stopping.
    weight_decay : float, default 5e-4
    verbose : bool, default True

    Returns
    -------
    nn.Module
        The trained model loaded with the best-AUC weights.
    """
    device = get_device()
    model = model.to(device)
    data = data.to(device)
    yt = data.y[data.train_mask]
    cw = min((yt == 0).sum().float() / max((yt == 1).sum().float(), 1), 10.0)
    w = torch.tensor([1.0, cw], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=w)
    best_auc, best_state, wait = 0.0, None, 0

    has_ts = hasattr(model, "time_enc")

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = (
            model(data.x, data.edge_index, ts=data.ts)
            if has_ts
            else model(data.x, data.edge_index)
        )
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            out = (
                model(data.x, data.edge_index, ts=data.ts)
                if has_ts
                else model(data.x, data.edge_index)
            )
            pr = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            yv = data.y[data.val_mask].cpu().numpy()
            auc = roc_auc_score(yv, pr) if len(np.unique(yv)) > 1 else 0.5
        if auc > best_auc:
            best_auc, best_state, wait = (
                float(auc),
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"  Early stopping ep {ep + 1}, best AUC: {best_auc:.4f}")
            break
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1:3d}  loss={loss.item():.4f}  val_auc={auc:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def eval_gnn(
    model: nn.Module,
    data,
    find_optimal_threshold,
    compute_metrics,
) -> tuple[dict, dict]:
    """Evaluate a trained GNN on val and test masks.

    Parameters
    ----------
    model : nn.Module
    data : torch_geometric.data.Data
    find_optimal_threshold : callable
        Function ``(y_true, y_proba) -> (threshold, f1)``.
    compute_metrics : callable
        Function ``(y_true, y_proba, threshold) -> dict``.

    Returns
    -------
    tuple of dict
        ``(val_metrics, test_metrics)``.
    """
    device = get_device()
    model.eval()
    data = data.to(device)
    has_ts = hasattr(model, "time_enc")
    with torch.no_grad():
        out = (
            model(data.x, data.edge_index, ts=data.ts)
            if has_ts
            else model(data.x, data.edge_index)
        )
        pv = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
        pt = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
    yv = data.y[data.val_mask].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()
    th, _ = find_optimal_threshold(yv, pv)
    return compute_metrics(yv, pv, th), compute_metrics(yt, pt, th)


def build_edge_index(edges_df) -> torch.Tensor:
    """Convert a two-column DataFrame of remapped txIds to a tensor."""
    src = edges_df.iloc[:, 0].to_numpy(dtype=np.int64)
    dst = edges_df.iloc[:, 1].to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()
