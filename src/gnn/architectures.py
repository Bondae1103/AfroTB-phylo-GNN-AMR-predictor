"""GNN architectures beyond the baseline MultiTaskGCN, plus a no-graph control.

All models here share one forward signature -- forward(x, edge_index,
edge_weight) -> logits -- so a single trainer can run any of them and the
comparison stays apples-to-apples.

What each is for
----------------
MLPOnly           No graph at all. The control. Every graph model must be
                  compared against this, because "the GNN scored well" means
                  nothing if an MLP on the same features scored the same.

MultiTaskGCN      The existing baseline (imported from model.py, unchanged).

GATv2Net          Dynamic attention over neighbours, with the Jaccard edge
                  weight passed as a 1-d edge FEATURE (edge_attr) rather than
                  hand-rolled into the attention logit -- GATv2Conv already
                  supports edge features, and reimplementing that arithmetic
                  would just be a less-tested copy of it.

GatedHybridGNN    Runs an MLP stream and a graph stream over the same input
                  and mixes them with a learned per-isolate gate
                  gamma = sigmoid(W_g x):

                      h = gamma * h_mlp + (1 - gamma) * h_gnn

                  This is primarily a MEASUREMENT INSTRUMENT, not a bid to win
                  the leaderboard. gamma is readable after training: if it
                  saturates near 1 the model has learned to ignore the graph,
                  which is a quantitative answer to "does phylogeny help?"
                  Report gamma alongside the score -- a hybrid that wins by
                  driving gamma to 1 has not shown that the graph helps, it
                  has shown the opposite.

Note on SAGEConv: an "ego-preserving GraphSAGE" is not a new architecture --
PyG's SAGEConv with root_weight=True (the default) already keeps a separate
transform for the central node. It is included here for completeness, with the
caveat that SAGEConv does not accept edge weights, so choosing it discards the
Jaccard weighting that the edge-weight ablation showed carries real signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

from .model import MultiTaskGCN  # noqa: F401  (re-exported for the registry)


class MLPOnly(nn.Module):
    """Graph-free control with the same trunk width as the GNNs."""

    def __init__(self, in_dim, hidden_dim, n_out, dropout=0.3, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, n_out)

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.head(self.net(x))

    def embed(self, x, edge_index=None, edge_weight=None):
        return self.net(x)


class GATv2Net(nn.Module):
    """2-layer GATv2 with the Jaccard weight as a 1-d edge feature."""

    def __init__(self, in_dim, hidden_dim, n_out, dropout=0.3,
                 heads=4, skip_connection=True, **_):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads,
                               dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1,
                               dropout=dropout, edge_dim=1)
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.head = nn.Linear(hidden_dim + (in_dim if skip_connection else 0), n_out)

    def _edge_attr(self, edge_index, edge_weight):
        if edge_weight is None:
            return None
        return edge_weight.view(-1, 1)

    def embed(self, x, edge_index, edge_weight):
        ea = self._edge_attr(edge_index, edge_weight)
        h = F.elu(self.conv1(x, edge_index, ea))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index, ea))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.skip_connection:
            h = torch.cat([h, x], dim=1)
        return h

    def forward(self, x, edge_index, edge_weight):
        return self.head(self.embed(x, edge_index, edge_weight))


class EgoSAGENet(nn.Module):
    """2-layer GraphSAGE. root_weight=True already separates self from neighbours.

    Discards edge_weight -- SAGEConv has no edge-weight argument. Kept explicit
    rather than silently ignored.
    """

    def __init__(self, in_dim, hidden_dim, n_out, dropout=0.3,
                 skip_connection=True, **_):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, root_weight=True)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, root_weight=True)
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.head = nn.Linear(hidden_dim + (in_dim if skip_connection else 0), n_out)

    def embed(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.skip_connection:
            h = torch.cat([h, x], dim=1)
        return h

    def forward(self, x, edge_index, edge_weight=None):
        return self.head(self.embed(x, edge_index, edge_weight))


class GatedHybridGNN(nn.Module):
    """MLP stream + GCN stream, mixed by a learned per-isolate gate.

    last_gate holds the most recent gamma (N, 1) after a forward pass, so the
    trainer can report its distribution. gamma -> 1 means "ignore the graph".
    """

    def __init__(self, in_dim, hidden_dim, n_out, dropout=0.3, **_):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gate = nn.Linear(in_dim, 1)
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, n_out)
        self.last_gate = None

    def embed(self, x, edge_index, edge_weight):
        h_mlp = self.mlp(x)
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_gnn = F.relu(self.conv2(h, edge_index, edge_weight))
        gamma = torch.sigmoid(self.gate(x))
        self.last_gate = gamma.detach()
        return gamma * h_mlp + (1.0 - gamma) * h_gnn

    def forward(self, x, edge_index, edge_weight):
        h = self.embed(x, edge_index, edge_weight)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    def gate_stats(self):
        """Summary of the learned gate. mean near 1.0 => the graph is unused."""
        if self.last_gate is None:
            return None
        g = self.last_gate.view(-1)
        return {
            "gamma_mean": float(g.mean()),
            "gamma_median": float(g.median()),
            "gamma_p05": float(torch.quantile(g, 0.05)),
            "gamma_p95": float(torch.quantile(g, 0.95)),
            "fraction_above_0.9": float((g > 0.9).float().mean()),
            "interpretation": "gamma is the weight on the graph-free MLP stream; "
                              "near 1.0 means the model learned to ignore the graph",
        }


class SkipGCN(nn.Module):
    """MultiTaskGCN's architecture, re-exposed with the shared signature.

    model.py's MultiTaskGCN is left untouched; this wrapper exists so the
    benchmark can treat every architecture uniformly (including calling
    .embed()) without editing working code.
    """

    def __init__(self, in_dim, hidden_dim, n_out, dropout=0.3,
                 skip_connection=True, **_):
        super().__init__()
        self.inner = MultiTaskGCN(in_dim, hidden_dim, n_out, dropout=dropout,
                                  skip_connection=skip_connection)

    @property
    def head(self):
        return self.inner.head

    def embed(self, x, edge_index, edge_weight):
        m = self.inner
        h = F.relu(m.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=m.dropout, training=self.training)
        h = F.relu(m.conv2(h, edge_index, edge_weight))
        h = F.dropout(h, p=m.dropout, training=self.training)
        if m.skip_connection:
            h = torch.cat([h, x], dim=1)
        return h

    def forward(self, x, edge_index, edge_weight):
        return self.inner(x, edge_index, edge_weight)


ARCHITECTURES = {
    "mlp_only": MLPOnly,
    "gcn_skip": SkipGCN,
    "gatv2": GATv2Net,
    "sage": EgoSAGENet,
    "gated_hybrid": GatedHybridGNN,
}

GRAPH_FREE = {"mlp_only"}
