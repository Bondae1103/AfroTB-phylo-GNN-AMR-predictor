"""PyTorch multi-task MLP for AMR prediction: shared trunk, 9 independent
drug logits in a single linear output layer (same hard-parameter-sharing
design as train_mlp.py's sklearn MLPClassifier and src/gnn/model.py's
MultiTaskGCN head, just with an explicit torch.nn implementation instead of
sklearn's).
"""

import torch.nn as nn


class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(128, 64), n_drugs=9, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_drugs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits, (N, n_drugs) -- sigmoid applied by the loss/eval, not here
