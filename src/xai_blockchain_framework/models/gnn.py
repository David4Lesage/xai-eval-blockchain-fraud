"""Graph Neural Network baselines (Temporal GCN and GraphSAGE).

The architecture mirrors the original research implementation that
produced the reported F1 of 0.88 on the Elliptic dataset:

- **TemporalGCN** uses two :class:`torch_geometric.nn.TAGConv` layers
  (K=3 then K=2) followed by a continuous time encoding concatenated to
  the node embeddings.
- **GraphSAGE** uses two :class:`torch_geometric.nn.SAGEConv` layers.

Both networks produce 2-class logits. Training follows the published
recipe: AUC-based early stopping, capped class weights, gradient
clipping, and dropout of 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from torch import nn
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
    """TAGConv-based temporal GCN with a continuous time encoding.

    Architecture (reproduces the original published recipe):

    - ``TAGConv(in_features -> hidden, K=3)`` + ReLU + dropout
    - ``TAGConv(hidden -> hidden, K=2)`` + ReLU + dropout
    - Concatenate a ``Linear(1 -> hidden)`` encoding of the normalized
      time step to the final node embedding.
    - ``Linear(2*hidden -> 2)`` classifier.

    Parameters
    ----------
    in_features : int
        Dimension of per-node input features.
    hidden : int, default 128
        Hidden channel size.
    out : int, default 2
        Number of output classes.
    K : int, default 3
        Filter size of the first TAGConv layer.
    dropout : float, default 0.5
        Dropout rate applied after each convolution.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 128,
        out: int = 2,
        K: int = 3,
        dropout: float = 0.5,
        n_timesteps: int | None = None,  # kept for API compatibility
    ) -> None:
        super().__init__()
        self.conv1 = TAGConv(in_features, hidden, K=K)
        self.conv2 = TAGConv(hidden, hidden, K=2)
        self.time_enc = nn.Linear(1, hidden)
        self.fc = nn.Linear(hidden * 2, out)
        self.dropout = dropout
        self.hidden = hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.dropout, self.training)
        if time_steps is not None:
            ts = time_steps.float()
            t_norm = (ts.unsqueeze(-1) - ts.min()) / (ts.max() - ts.min() + 1e-8)
            h = torch.cat([h, self.time_enc(t_norm)], dim=-1)
        else:
            h = torch.cat(
                [h, torch.zeros(h.size(0), self.hidden, device=h.device)], dim=-1,
            )
        return self.fc(h)


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE (mean aggregation, dropout 0.5).

    Parameters
    ----------
    in_features : int
        Dimension of per-node input features.
    hidden : int, default 128
        Hidden channel size.
    out : int, default 2
        Number of output classes.
    dropout : float, default 0.5
        Dropout rate applied after each convolution.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 128,
        out: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.fc = nn.Linear(hidden, out)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.dropout, self.training)
        return self.fc(h)


@dataclass
class GNNTrainingResult:
    """Training trace returned by :func:`train_gnn`."""

    best_val_auc: float
    best_epoch: int
    history: list[dict[str, float]]


def train_gnn(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    time_steps: torch.Tensor | None = None,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    class_weight_cap: float = 10.0,
    grad_clip: float = 1.0,
    patience: int = 30,
    verbose: bool = True,
) -> GNNTrainingResult:
    """Train a GNN with AUC-based early stopping and capped class weights.

    Mirrors the original research recipe that produced F1 ~0.88 on the
    Elliptic test split.

    Parameters
    ----------
    model : nn.Module
        The GNN instance to train. Must accept ``(x, edge_index)`` and
        optionally ``time_steps`` via the keyword argument used by
        :class:`TemporalGCN`.
    x, edge_index, y : torch.Tensor
        Node features, edge index, and node labels.
    train_mask, val_mask : torch.Tensor
        Boolean masks for train and validation nodes.
    time_steps : torch.Tensor, optional
        Required for :class:`TemporalGCN`.
    epochs : int, default 200
        Maximum number of training epochs.
    lr : float, default 1e-2
        Learning rate.
    weight_decay : float, default 5e-4
        L2 regularization.
    class_weight_cap : float, default 10.0
        Cap on the positive-class weight used in the cross-entropy loss.
        ``w_pos = min(n_neg / n_pos, class_weight_cap)``.
    grad_clip : float, default 1.0
        Max norm for gradient clipping (set to 0 to disable).
    patience : int, default 30
        Epochs without AUC improvement before early stopping.
    verbose : bool, default True
        Print progress every 10 epochs.

    Returns
    -------
    GNNTrainingResult
    """
    device = next(model.parameters()).device
    has_time_enc = hasattr(model, "time_enc")

    yt = y[train_mask]
    n_pos = (yt == 1).sum().float()
    n_neg = (yt == 0).sum().float()
    cw = min(float(n_neg / max(n_pos.item(), 1)), class_weight_cap)
    class_weights = torch.tensor([1.0, cw], device=device, dtype=torch.float32)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float]] = []

    if verbose:
        print(f"  class weights: [1.0, {cw:.2f}]  (n_pos={int(n_pos.item())}, n_neg={int(n_neg.item())})")

    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = (
            model(x, edge_index, time_steps=time_steps)
            if has_time_enc
            else model(x, edge_index)
        )
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        model.eval()
        with torch.no_grad():
            logits_eval = (
                model(x, edge_index, time_steps=time_steps)
                if has_time_enc
                else model(x, edge_index)
            )
            probs_val = F.softmax(logits_eval[val_mask], dim=1)[:, 1].cpu().numpy()
            y_val = y[val_mask].cpu().numpy()
            if len(np.unique(y_val)) > 1:
                auc = float(roc_auc_score(y_val, probs_val))
            else:
                auc = 0.5

        history.append({"epoch": epoch, "loss": float(loss.item()), "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  val_auc={auc:.4f}")
        if wait >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best epoch {best_epoch}, auc={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return GNNTrainingResult(best_val_auc=best_auc, best_epoch=best_epoch, history=history)


def build_edge_index(edges_df) -> torch.Tensor:
    """Convert an edges DataFrame with columns ``txId1, txId2`` to a tensor.

    Assumes both ids have already been remapped to dense integers aligned
    with the feature matrix rows.
    """
    src = edges_df.iloc[:, 0].to_numpy(dtype=np.int64)
    dst = edges_df.iloc[:, 1].to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()
