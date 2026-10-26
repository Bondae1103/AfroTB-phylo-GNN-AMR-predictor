"""Multi-task GCN for AMR prediction: a shared 2-layer GCN encoder + one
linear multi-label output layer (9 logits, one per drug).

This is the graph analogue of src/baselines/train_mlp.py's hard-parameter-
sharing multi-task MLP: shared representation-learning layers (here, graph
convolutions that mix each isolate's own 157-mutation features with its
phylogenetic neighbors', weighted by Jaccard-derived edge_weight) feeding
independent per-drug output units, rather than 9 separate models.

skip_connection=True concatenates the isolate's own raw feature
vector onto the final GCN hidden state before the head, so the head sees
both graph-smoothed context AND the isolate's own unsmoothed signal
directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MultiTaskGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_drugs=9, n_out=None, dropout=0.3,
                 skip_connection=True, **_):
        super().__init__()
        n_outputs = n_out if n_out is not None else n_drugs
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.skip_connection = skip_connection
        head_in_dim = hidden_dim + in_dim if skip_connection else hidden_dim
        self.head = nn.Linear(head_in_dim, n_outputs)

    def embed(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.skip_connection:
            h = torch.cat([h, x], dim=1)
        return h

    def forward(self, x, edge_index, edge_weight=None):
        return self.head(self.embed(x, edge_index, edge_weight))
