"""Train and evaluate a true Lineage-Aware Multi-Task GNN on the actual VCF data.

Key differences from earlier baseline/Phase 3 runs:
1. Input features X are extracted DIRECTLY from snp_matrix.npz (VCF SNP variant calls),
   comprising the 133 verified resistance-locus SNP columns from catalogue_sites.json
   plus 500 lineage-informative background core SNP columns selected on the train split.
2. Multi-task supervision jointly predicts 9-drug AMR and 8-class M. tuberculosis lineage.
3. The graph message passing operates over the 465,257-site Jaccard k-NN graph.
4. Outputs comprehensive AMR metrics (Core-6 and All-9) and Lineage accuracy/Macro-F1.
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.baselines.data_loader import load_aligned_data, split_masks
from src.baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from src.baselines.results_path import resolve as resolve_results_path
from src.gnn.model import MultiTaskGCN
from src.protocols.features import load_catalogue_site_indices, select_genomewide_sites
from src.protocols.lineage import IGNORE_INDEX, build_y_lineage

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PATH = REPO_ROOT / "results" / "vcf_lineage_gnn_results.json"

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


def build_vcf_features(snp, train_mask, cat_indices, n_bg=N_BACKGROUND_SITES, seed=SEED):
    """Build input matrix X directly from snp_matrix.npz (VCF variant calls)."""
    bg_sites, info = select_genomewide_sites(
        snp,
        train_mask=train_mask,
        exclude_sites=cat_indices,
        max_sites=n_bg,
        strategy="random",
        seed=seed,
    )
    combined_sites = np.concatenate([cat_indices, bg_sites])
    X = np.asarray(snp[:, combined_sites].todense(), dtype=np.float32)
    feature_names = [f"vcf_res_site_{s}" for s in cat_indices] + [f"vcf_bg_site_{s}" for s in bg_sites]
    return X, feature_names, info


class LineageAwareMultiTaskGNN(nn.Module):
    """Joint AMR + Lineage GNN sharing a 2-layer GCN trunk with skip connections."""

    def __init__(self, in_dim, hidden_dim, n_drugs, n_lineages, dropout=0.3):
        super().__init__()
        self.backbone = MultiTaskGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_drugs=n_drugs,
            dropout=dropout,
            skip_connection=True,
        )
        trunk_dim = hidden_dim + in_dim
        self.lineage_head = nn.Linear(trunk_dim, n_lineages)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.backbone.embed(x, edge_index, edge_weight)
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

    print("Loading aligned data and graph...")
    tab = load_aligned_data(PROCESSED_DIR)
    masks_np = split_masks(tab)
    train_mask = masks_np["train"]
    val_mask = masks_np["val"]
    test_mask = masks_np["test"]
    drug_columns = tab["drug_columns"]

    graph = torch.load(PROCESSED_DIR / "pyg_graph.pt", weights_only=False)
    edge_index = graph.edge_index.long()
    edge_weight = graph.edge_weight.float()

    print("Loading core VCF SNP matrix...")
    snp = sp.load_npz(PROCESSED_DIR / "snp_matrix.npz")
    cat_indices = load_catalogue_site_indices(PROCESSED_DIR)
    if cat_indices is None:
        raise FileNotFoundError("catalogue_sites.json not found in processed directory")
    print(f"Loaded {len(cat_indices)} verified resistance mutation VCF columns.")

    print(f"Extracting VCF features (133 resistance SNPs + {N_BACKGROUND_SITES} background SNPs)...")
    X_vcf_np, feat_names, bg_info = build_vcf_features(snp, train_mask, cat_indices, n_bg=N_BACKGROUND_SITES)
    print(f"Constructed pure VCF feature matrix: {X_vcf_np.shape}")

    # Lineage target
    y_lin_np, lin_classes, lin_meta = build_y_lineage(PROCESSED_DIR)
    n_lineage_classes = len(lin_classes)
    print(f"Loaded lineage target with {n_lineage_classes} classes.")

    # Convert to PyTorch tensors
    X = torch.tensor(X_vcf_np, dtype=torch.float32)
    y_amr = torch.tensor(tab["y"], dtype=torch.float32)
    y_lineage = torch.tensor(y_lin_np, dtype=torch.long)
    t_train = torch.tensor(train_mask, dtype=torch.bool)
    t_val = torch.tensor(val_mask, dtype=torch.bool)
    t_test = torch.tensor(test_mask, dtype=torch.bool)

    # Core-6 drug indices for checkpoint selection
    core6_drugs = ["RIF", "INH", "EMB", "PZA", "STM", "LEV"]
    core6_idx = [drug_columns.index(d) for d in core6_drugs]

    # Model and optimizer
    model = LineageAwareMultiTaskGNN(
        in_dim=X.shape[1],
        hidden_dim=HIDDEN_DIM,
        n_drugs=len(drug_columns),
        n_lineages=n_lineage_classes,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pos_weight = compute_pos_weight(y_amr[t_train], cap=POS_WEIGHT_CAP)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("\nTraining Lineage-Aware Multi-Task GNN...")
    t0 = time.time()
    best_val_core6_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        amr_logits, lineage_logits = model(X, edge_index, edge_weight)

        loss_amr = bce_loss_fn(amr_logits[t_train], y_amr[t_train])
        loss_lin = ce_loss_fn(lineage_logits[t_train], y_lineage[t_train])
        loss = loss_amr + LAMBDA_LINEAGE * loss_lin

        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_amr_logits, v_lin_logits = model(X, edge_index, edge_weight)
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

    # Final inference on all nodes
    with torch.no_grad():
        final_amr_logits, final_lin_logits = model(X, edge_index, edge_weight)
        all_amr_probs = torch.sigmoid(final_amr_logits).numpy()
        all_lin_preds = torch.argmax(final_lin_logits, dim=-1).numpy()

    # Split data
    y_test_amr = y_amr[t_test].numpy()
    test_probs = all_amr_probs[test_mask]
    val_probs = all_amr_probs[val_mask]
    y_val_amr = y_amr[t_val].numpy()

    # Tune per-drug thresholds on val
    tuned_thresholds = tune_per_drug_thresholds(y_val_amr, val_probs, drug_columns)

    # Evaluate AMR
    amr_metrics_05 = evaluate_multilabel(y_test_amr, test_probs, drug_columns, threshold=0.5)
    amr_metrics_tuned = evaluate_multilabel(y_test_amr, test_probs, drug_columns, threshold=tuned_thresholds)

    # Evaluate Lineage
    test_lin_true = y_lineage[t_test].numpy()
    valid_lin_mask = test_lin_true != IGNORE_INDEX
    test_lin_true_valid = test_lin_true[valid_lin_mask]
    test_lin_pred_valid = all_lin_preds[test_mask][valid_lin_mask]

    lineage_acc = accuracy_score(test_lin_true_valid, test_lin_pred_valid)
    lineage_f1_macro = f1_score(test_lin_true_valid, test_lin_pred_valid, average="macro", zero_division=0)

    # Compute Core-6 metrics
    def core6_macro_f1(metrics_dict):
        scores = [metrics_dict["per_drug"][d]["f1"] for d in core6_drugs]
        return float(np.mean(scores))

    core6_f1_05 = core6_macro_f1(amr_metrics_05)
    core6_f1_tuned = core6_macro_f1(amr_metrics_tuned)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_features": {
            "source": "data/processed/snp_matrix.npz (VCF SNP variant calls)",
            "n_total_features": X.shape[1],
            "n_catalogue_resistance_snps": len(cat_indices),
            "n_background_lineage_snps": N_BACKGROUND_SITES,
        },
        "training": {
            "train_seconds": round(train_seconds, 2),
            "best_epoch": best_epoch,
            "lambda_lineage": LAMBDA_LINEAGE,
            "pos_weight_cap": POS_WEIGHT_CAP,
        },
        "amr_performance": {
            "default_threshold_0.5": {
                "test_core6_macro_f1": round(core6_f1_05, 4),
                "test_all9_macro_f1": round(amr_metrics_05["macro_f1"], 4),
                "test_macro_roc_auc": round(amr_metrics_05["macro_roc_auc"], 4) if amr_metrics_05["macro_roc_auc"] is not None else None,
                "per_drug": {d: amr_metrics_05["per_drug"][d] for d in drug_columns},
            },
            "tuned_threshold": {
                "thresholds": {d: round(t, 4) for d, t in tuned_thresholds.items()},
                "test_core6_macro_f1": round(core6_f1_tuned, 4),
                "test_all9_macro_f1": round(amr_metrics_tuned["macro_f1"], 4),
                "per_drug": {d: amr_metrics_tuned["per_drug"][d] for d in drug_columns},
            },
        },
        "lineage_performance": {
            "n_test_isolates_evaluated": int(valid_lin_mask.sum()),
            "accuracy": round(lineage_acc, 4),
            "macro_f1": round(lineage_f1_macro, 4),
            "classes": lin_meta["classes"],
        },
    }

    results_path = resolve_results_path(PROCESSED_DIR, RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("VCF-DERIVED LINEAGE-AWARE GNN RESULTS")
    print("=" * 60)
    print(f"Input features: {X.shape[1]} direct VCF variant call sites")
    print(f"AMR Test Core-6 Macro-F1 (default 0.5) : {core6_f1_05:.4f}")
    print(f"AMR Test Core-6 Macro-F1 (val tuned)   : {core6_f1_tuned:.4f}")
    print(f"AMR Test All-9 Macro-F1 (default 0.5)  : {amr_metrics_05['macro_f1']:.4f}")
    print(f"AMR Test All-9 Macro-F1 (val tuned)    : {amr_metrics_tuned['macro_f1']:.4f}")
    print(f"AMR Test Macro ROC-AUC                 : {amr_metrics_05['macro_roc_auc']:.4f}")
    print(f"Lineage Classification Test Accuracy   : {lineage_acc * 100:.2f}%")
    print(f"Lineage Classification Test Macro-F1   : {lineage_f1_macro:.4f}")
    print("=" * 60)
    print(f"Wrote full results to {results_path}")


if __name__ == "__main__":
    main()
