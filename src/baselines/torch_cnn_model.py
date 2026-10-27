"""PyTorch multi-task 1D-CNN for AMR prediction: the (N, F) binary mutation
vector is treated as a length-F single-channel sequence, passed through
Conv1d blocks, and read out by a shared MLP head into 9 independent drug
logits (same multi-task output design as torch_mlp_model.py and
src/gnn/model.py, so the three are comparable).

Caveat: the 157 catalogue columns are ordered by the catalogue, not by genome
position, so convolutional locality carries no biological meaning on
features.csv. On genome-ordered SNP inputs (snp_matrix.npz) it does: those
columns are sorted by (pos, alt). Run it on them through the protocol benchmark
as architecture "cnn1d" (see src/experiments/benchmark.py), e.g.

    .venv/Scripts/python.exe -m src.experiments.run_benchmark \
        --architectures cnn1d --protocols genomewide --seeds 0 1 2 \
        --out results/protocol_benchmark_cnn.json

Always pass --out: the default is results/protocol_benchmark.json, which holds
the committed GNN/baseline grid.
"""

import torch.nn as nn


class TorchCNN(nn.Module):
    def __init__(self, in_dim, channels=(32, 64), kernel_size=5, hidden_dim=128, n_drugs=9, dropout=0.3, pool=2):
        super().__init__()
        blocks = []
        prev, length = 1, in_dim
        for c in channels:
            blocks += [
                nn.Conv1d(prev, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(pool),
            ]
            prev, length = c, length // pool
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev * length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_drugs),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))  # (N, F) -> (N, 1, F) -> raw logits (N, n_drugs)
