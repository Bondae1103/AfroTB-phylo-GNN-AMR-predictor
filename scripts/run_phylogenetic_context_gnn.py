"""Train and evaluate the Full Phylogeny-Aware GNN on real Afro-TB data.

Integrates complete phylogenetic and phylogeographic context:
1. Clonal MST tree backbone + transmission cluster network (from phylo_tree_graph.pt).
2. Pure VCF variant call input features (133 resistance SNPs + 500 background SNPs from snp_matrix.npz).
3. Learned Lineage embeddings (16 dims, 8 lineages).
4. Learned Phylogeographic Country embeddings (8 dims, 26 African countries).
5. Joint Multi-Task supervision for 9-drug AMR and 8-class Lineage.
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.baselines.data_loader import load_aligned_data, split_masks
from src.baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from src.baselines.results_path import resolve as resolve_results_path
from src.gnn.model import MultiTaskGCN
from src.protocols.features import load_catalogue_site_indices, select_genomewide_sites
from src.protocols.lineage import IGNORE_INDEX, build_y_lineage
from src.protocols.metadata_features import load_metadata_features

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PATH = REPO_ROOT / "results" / "phylo_context_gnn_results.json"
GRAPH_PATH = PROCESSED_DIR / "phylo_tree_graph.pt"

SEED = 42
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.02
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 250
PATIENCE = 20
LAMBDA_LINEAGE = 0.5
POS_WEIGHT_CAP = 20.0
N_BACKGROUND_SITES = 500
LINEAGE_EMB_DIM = 16
COUNTRY_EMB_DIM = 8


class PhylogenyContextGNN(nn.Module):
    """Integrates VCF variants, phylogenetic graph message passing, and metadata embeddings."""

    def __init__(self, vcf_dim, n_lineage_tokens, n_country_tokens, hidden_dim, n_drugs, n_lineage_classes, dropout=0.3):
        super().__init__()
        self.lineage_emb = nn.Embedding(n_lineage_tokens, LINEAGE_EMB_DIM, padding_idx=0)
        self.country_emb = nn.Embedding(n_country_tokens, COUNTRY_EMB_DIM, padding_idx=0)

        in_dim = vcf_dim + LINEAGE_EMB_DIM + COUNTRY_EMB_DIM
        self.backbone = MultiTaskGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_drugs=n_drugs,
            dropout=dropout,
            skip_connection=True,
        )
        trunk_dim = hidden_dim + in_dim
        self.lineage_head = nn.Linear(trunk_dim, n_lineage_classes)

    def forward(self, x_vcf, lin_idx, cty_idx, edge_index, edge_weight=None):
        e_lin = self.lineage_emb(lin_idx)
        e_cty = self.country_emb(cty_idx)
        x_combined = torch.cat([x_vcf, e_lin, e_cty], dim=1)

        h = self.backbone.embed(x_combined, edge_index, edge_weight)
        amr_logits = self.backbone.head(h)
        lineage_logits = self.lineage_head(h)
        return amr_logits, lineage_logits


def compute_pos_weight(y_train, cap=POS_WEIGHT_CAP):
    n_pos = y_train.sum(dim=0)
    n_neg = y_train.shape[0] - n_pos
    pos_weight = torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1), torch.ones_like(n_pos))
    if cap is not None:
        pos_weight = torch.clamp(pos_weight, max=cap)
    return pos_weight


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading tabular alignment and data...")
    tab = load_aligned_data(PROCESSED_DIR)
    masks_np = split_masks(tab)
    train_mask, val_mask, test_mask = masks_np["train"], masks_np["val"], masks_np["test"]
    drug_columns = tab["drug_columns"]

    print(f"Loading phylogenetic network from {GRAPH_PATH.name}...")
    graph = torch.load(GRAPH_PATH, weights_only=False)
    edge_index = graph.edge_index.long()
    edge_weight = graph.edge_weight.float()
    print(f"Loaded phylogenetic graph with {edge_index.shape[1]} directed edges across {len(graph.sample_ids)} isolates.")

    print("Loading VCF core SNP matrix...")
    snp = sp.load_npz(PROCESSED_DIR / "snp_matrix.npz")
    cat_indices = load_catalogue_site_indices(PROCESSED_DIR)
    if cat_indices is None:
        raise FileNotFoundError("catalogue_sites.json not found in processed directory")

    print(f"Selecting {N_BACKGROUND_SITES} lineage background SNPs on train rows...")
    bg_sites, _ = select_genomewide_sites(
        snp, train_mask=train_mask, exclude_sites=cat_indices,
        max_sites=N_BACKGROUND_SITES, strategy="random", seed=SEED
    )
    combined_sites = np.concatenate([cat_indices, bg_sites])
    X_vcf_np = np.asarray(snp[:, combined_sites].todense(), dtype=np.float32)
    print(f"Constructed VCF variant feature matrix: {X_vcf_np.shape}")

    print("Loading Lineage and Country metadata...")
    meta = load_metadata_features(PROCESSED_DIR)
    lin_indices = meta["lineage_indices"]
    cty_indices = meta["country_indices"]
    n_lin_tokens = meta["info"]["n_lineage_classes"] + 1  # 0 is unassigned
    n_cty_tokens = meta["info"]["n_countries"] + 1        # 0 is unassigned

    # Target lineage classification
    y_lin_np, lin_classes, _ = build_y_lineage(PROCESSED_DIR)
    n_lineage_classes = len(lin_classes)

    # Convert to PyTorch tensors
    X_vcf = torch.tensor(X_vcf_np, dtype=torch.float32)
    y_amr = torch.tensor(tab["y"], dtype=torch.float32)
    y_lineage = torch.tensor(y_lin_np, dtype=torch.long)
    t_train = torch.tensor(train_mask, dtype=torch.bool)
    t_val = torch.tensor(val_mask, dtype=torch.bool)
    t_test = torch.tensor(test_mask, dtype=torch.bool)

    core6_drugs = ["RIF", "INH", "EMB", "PZA", "STM", "LEV"]
    core6_idx = [drug_columns.index(d) for d in core6_drugs]

    # Initialize model
    model = PhylogenyContextGNN(
        vcf_dim=X_vcf.shape[1],
        n_lineage_tokens=n_lin_tokens,
        n_country_tokens=n_cty_tokens,
        hidden_dim=HIDDEN_DIM,
        n_drugs=len(drug_columns),
        n_lineage_classes=n_lineage_classes,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pos_weight = compute_pos_weight(y_amr[t_train], cap=POS_WEIGHT_CAP)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("\nTraining Full Phylogeny-Aware GNN...")
    t0 = time.time()
    best_val_core6_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        amr_logits, lineage_logits = model(X_vcf, lin_indices, cty_indices, edge_index, edge_weight)

        loss_amr = bce_loss_fn(amr_logits[t_train], y_amr[t_train])
        loss_lin = ce_loss_fn(lineage_logits[t_train], y_lineage[t_train])
        loss = loss_amr + LAMBDA_LINEAGE * loss_lin

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_amr_logits, _ = model(X_vcf, lin_indices, cty_indices, edge_index, edge_weight)
            val_probs = torch.sigmoid(v_amr_logits[t_val]).numpy()
            val_true = y_amr[t_val].numpy()
            val_preds = (val_probs[:, core6_idx] >= 0.5).astype(int)
            val_core6_f1 = f1_score(val_true[:, core6_idx], val_preds, average="macro", zero_division=0)

        if val_core6_f1 > best_val_core6_f1:
            best_val_core6_f1 = val_core6_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f} (AMR={loss_amr.item():.4f}, Lin={loss_lin.item():.4f}) | Val Core-6 F1={val_core6_f1:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val Core-6 F1: {best_val_core6_f1:.4f})")
            break

    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        final_amr_logits, final_lin_logits = model(X_vcf, lin_indices, cty_indices, edge_index, edge_weight)
        all_amr_probs = torch.sigmoid(final_amr_logits).numpy()
        all_lin_preds = torch.argmax(final_lin_logits, dim=-1).numpy()

    y_test_amr = y_amr[t_test].numpy()
    test_probs = all_amr_probs[test_mask]
    val_probs = all_amr_probs[val_mask]
    y_val_amr = y_amr[t_val].numpy()

    tuned_thresholds = tune_per_drug_thresholds(y_val_amr, val_probs, drug_columns)
    amr_metrics_05 = evaluate_multilabel(y_test_amr, test_probs, drug_columns, threshold=0.5)
    amr_metrics_tuned = evaluate_multilabel(y_test_amr, test_probs, drug_columns, threshold=tuned_thresholds)

    # Lineage evaluation
    test_lin_true = y_lineage[t_test].numpy()
    valid_lin_mask = test_lin_true != IGNORE_INDEX
    lineage_acc = accuracy_score(test_lin_true[valid_lin_mask], all_lin_preds[test_mask][valid_lin_mask])
    lineage_f1_macro = f1_score(test_lin_true[valid_lin_mask], all_lin_preds[test_mask][valid_lin_mask], average="macro", zero_division=0)

    def core6_macro_f1(metrics_dict):
        return float(np.mean([metrics_dict["per_drug"][d]["f1"] for d in core6_drugs]))

    core6_f1_05 = core6_macro_f1(amr_metrics_05)
    core6_f1_tuned = core6_macro_f1(amr_metrics_tuned)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phylogenetic_context": {
            "graph_type": "Clonal MST backbone + transmission clusters",
            "n_directed_edges": edge_index.shape[1],
            "n_vcf_features": X_vcf.shape[1],
            "lineage_embedding_dim": LINEAGE_EMB_DIM,
            "country_embedding_dim": COUNTRY_EMB_DIM,
        },
        "training": {
            "train_seconds": round(train_seconds, 2),
            "best_epoch": best_epoch,
            "lambda_lineage": LAMBDA_LINEAGE,
        },
        "amr_performance": {
            "default_0.5": {
                "test_core6_macro_f1": round(core6_f1_05, 4),
                "test_all9_macro_f1": round(amr_metrics_05["macro_f1"], 4),
                "test_macro_roc_auc": round(amr_metrics_05["macro_roc_auc"], 4) if amr_metrics_05["macro_roc_auc"] else None,
                "per_drug": {d: amr_metrics_05["per_drug"][d] for d in drug_columns},
            },
            "tuned_threshold": {
                "test_core6_macro_f1": round(core6_f1_tuned, 4),
                "test_all9_macro_f1": round(amr_metrics_tuned["macro_f1"], 4),
                "thresholds": {d: round(t, 4) for d, t in tuned_thresholds.items()},
                "per_drug": {d: amr_metrics_tuned["per_drug"][d] for d in drug_columns},
            },
        },
        "lineage_performance": {
            "test_accuracy": round(lineage_acc, 4),
            "test_macro_f1": round(lineage_f1_macro, 4),
            "classes": lin_classes,
        },
    }

    results_path = resolve_results_path(PROCESSED_DIR, RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("PHYLOGENY-AWARE MULTI-TASK GNN RESULTS")
    print("=" * 65)
    print(f"Graph topology: Clonal MST Backbone + Outbreaks ({edge_index.shape[1]} edges)")
    print(f"Features: VCF Variants (633) + Lineage (16d) + Country (8d)")
    print(f"AMR Test Core-6 Macro-F1 (default 0.5) : {core6_f1_05:.4f}")
    print(f"AMR Test Core-6 Macro-F1 (val tuned)   : {core6_f1_tuned:.4f}")
    print(f"AMR Test All-9 Macro-F1 (default 0.5)  : {amr_metrics_05['macro_f1']:.4f}")
    print(f"AMR Test All-9 Macro-F1 (val tuned)    : {amr_metrics_tuned['macro_f1']:.4f}")
    print(f"AMR Test Macro ROC-AUC                 : {amr_metrics_05['macro_roc_auc']:.4f}")
    print(f"Lineage Classification Test Accuracy   : {lineage_acc * 100:.2f}%")
    print(f"Lineage Classification Test Macro-F1   : {lineage_f1_macro:.4f}")
    print("=" * 65)
    print(f"Saved full results to {results_path}")


if __name__ == "__main__":
    main()
