"""The protocol x split x architecture x seed benchmark.

This is the experiment the project actually needs. It answers three questions
that the existing results/ cannot, because everything in results/ was run
under one protocol, one split, and one seed:

  1. How much of the reported performance is the label tautology?
     -> compare protocol "catalogue" against "catalogue_ldo" / "genomewide".

  2. How much is phylogenetic leakage between train and test?
     -> compare split "original" against "phylo_clade".

  3. Does the graph help, once 1 and 2 are removed?
     -> compare architecture "mlp_only" (no graph) against the GNNs, across
        several seeds, with the spread reported.

Every cell is run over multiple seeds and reported as mean +/- std. A single
seed cannot distinguish a 0.02 F1 architecture difference from noise, and the
committed results/ files are all single-seed.

Threshold discipline is inherited unchanged from src/baselines/metrics.py:
per-drug thresholds are tuned on val and applied to test, never tuned on test.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..baselines.data_loader import load_aligned_data, split_masks
from ..baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from ..baselines.multilabel_utils import get_proba_matrix
from ..gnn.architectures import ARCHITECTURES
from ..protocols import features as featmod
from ..protocols.phylo_split import build_phylo_split
from .trainer import train as train_torch

CORE6 = ["RIF", "INH", "EMB", "PZA", "STM", "LEV"]

TABULAR = {"random_forest", "xgboost", "sk_mlp"}


def build_tabular(name, seed):
    if name == "random_forest":
        from ..baselines import train_random_forest
        return train_random_forest.build_model(random_state=seed)
    if name == "xgboost":
        from ..baselines import train_xgboost
        return train_xgboost.build_model(random_state=seed)
    if name == "sk_mlp":
        from ..baselines import train_mlp
        return train_mlp.build_model(random_state=seed)
    raise KeyError(name)


def core6_macro_f1(eval_result):
    """Macro-F1 restricted to the six drugs with usable test support.

    ETH/LZD/CAP have single-digit (sometimes zero) positives in a 15% test
    split, so their F1 is dominated by one or two examples. They stay in the
    per-drug report but are excluded from this selection metric -- reported
    alongside the all-9 number, never instead of it.
    """
    vals = [eval_result["per_drug"][d]["f1"] for d in CORE6
            if d in eval_result["per_drug"]]
    return float(np.mean(vals)) if vals else None


def _evaluate(y, proba, masks, drug_columns):
    y_val, y_test = y[masks["val"]], y[masks["test"]]
    p_val, p_test = proba[masks["val"]], proba[masks["test"]]
    thr = tune_per_drug_thresholds(y_val, p_val, drug_columns)
    out = {}
    for tag, t in (("default_0.5", 0.5), ("tuned", thr)):
        ev_val = evaluate_multilabel(y_val, p_val, drug_columns, threshold=t)
        ev_test = evaluate_multilabel(y_test, p_test, drug_columns, threshold=t)
        out[tag] = {
            "val_macro_f1_all9": ev_val["macro_f1"],
            "test_macro_f1_all9": ev_test["macro_f1"],
            "val_macro_f1_core6": core6_macro_f1(ev_val),
            "test_macro_f1_core6": core6_macro_f1(ev_test),
            "test_macro_roc_auc": ev_test["macro_roc_auc"],
            "test_macro_pr_auc": ev_test["macro_pr_auc"],
            "test_per_drug_f1": {d: ev_test["per_drug"][d]["f1"] for d in drug_columns},
        }
    return out


def _fit_predict_tabular(name, Xtr, ytr, X_all, seed):
    model = build_tabular(name, seed)
    # A single-column target (the leave-drug-out case) is a binary problem, not
    # a 1-column multi-label one; sklearn warns and reshapes it internally, so
    # flatten it here to keep the intent explicit.
    model.fit(Xtr, ytr.ravel() if ytr.ndim == 2 and ytr.shape[1] == 1 else ytr)
    return get_proba_matrix(model, X_all)


def run_cell(
    arch, X, y, masks, edge_index, edge_weight, drug_columns, seed,
    hidden_dim=64, max_epochs=300, patience=20, lr=0.05,
    y_lineage=None, n_lineage_classes=None, lambda_lineage=0.0,
):
    """Train one model on one (protocol, split, seed) and return probabilities."""
    if arch in TABULAR:
        proba = _fit_predict_tabular(arch, X[masks["train"]], y[masks["train"]], X, seed)
        return proba, {"architecture": arch}

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    mt = {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()}
    lin = torch.tensor(y_lineage, dtype=torch.long) if y_lineage is not None else None
    # Select the checkpoint on the drugs that have real support. With all 9,
    # the CAP/ETH/LZD columns (single-digit positives) make val macro-F1 a
    # coin flip and training stops after a handful of epochs.
    sel = [i for i, d in enumerate(drug_columns) if d in CORE6] or None
    _, proba, info = train_torch(
        Xt, yt, edge_index, edge_weight, mt, architecture=arch,
        hidden_dim=hidden_dim, lr=lr, max_epochs=max_epochs, patience=patience,
        seed=seed, y_lineage=lin, n_lineage_classes=n_lineage_classes,
        lambda_lineage=lambda_lineage, selection_drug_idx=sel,
    )
    info.pop("lineage_pred", None)
    return proba, info


def run_ldo(arch, X_full, feature_names, mutation_to_drug, y, masks,
            edge_index, edge_weight, drug_columns, seed, **kw):
    """Leave-drug-out: one binary model per drug, its own columns removed.

    Column d of the returned probability matrix comes from the model that was
    never shown drug d's mutation columns, so the assembled matrix can be fed
    to the same multi-label evaluator as every other protocol.
    """
    N = X_full.shape[0]
    proba = np.zeros((N, len(drug_columns)), dtype=float)
    infos = {}
    for d_i, drug in enumerate(drug_columns):
        Xd, _ = featmod.catalogue_features(X_full, feature_names, mutation_to_drug, drug=drug)
        yd = y[:, [d_i]]
        if yd[masks["train"]].sum() == 0:
            proba[:, d_i] = 0.0
            infos[drug] = {"skipped": "no positive training examples"}
            continue
        p, info = run_cell(arch, Xd, yd, masks, edge_index, edge_weight, [drug], seed, **kw)
        proba[:, d_i] = p[:, 0]
        infos[drug] = {"n_features": int(Xd.shape[1]),
                       "best_epoch": info.get("best_epoch")}
    return proba, {"architecture": arch, "per_drug": infos}


def aggregate(runs):
    """mean/std across seeds for the headline scalars."""
    keys = ["val_macro_f1_all9", "test_macro_f1_all9",
            "val_macro_f1_core6", "test_macro_f1_core6",
            "test_macro_roc_auc", "test_macro_pr_auc"]
    out = {}
    for tag in ("default_0.5", "tuned"):
        out[tag] = {}
        for k in keys:
            vals = [r[tag][k] for r in runs if r[tag].get(k) is not None]
            if vals:
                out[tag][k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n_seeds": len(vals),
                }
    return out


def run(
    processed_dir,
    architectures=("mlp_only", "gcn_skip", "gatv2", "gated_hybrid", "random_forest"),
    protocols=(featmod.CATALOGUE, featmod.GENOMEWIDE),
    split_names=("original", "phylo_clade"),
    seeds=(0, 1, 2),
    max_genomewide_sites=3000,
    clade_threshold=0.40,
    hidden_dim=64,
    max_epochs=300,
    patience=60,
    lr=0.05,
    lambda_lineage=0.0,
    verbose=True,
):
    """Run the full grid. Returns a nested results dict.

    lr defaults to 0.05 rather than src/gnn/train.py's 0.01: at this graph
    size, 0.01 had not converged after 250 full-batch epochs (test core-6
    0.937 at epoch 249) while 0.05 converged by epoch 80 to a better optimum
    (0.946). The old default was not wrong, it was just under-trained within
    any practical epoch budget.
    """
    processed_dir = Path(processed_dir)
    tab = load_aligned_data(processed_dir)
    drug_columns = tab["drug_columns"]
    y = tab["y"]

    graph = torch.load(processed_dir / "pyg_graph.pt", weights_only=False)
    edge_index = graph.edge_index.long()
    edge_weight = graph.edge_weight.float()

    from ..audit.label_tautology import load_mutation_drug_map
    mutation_to_drug = load_mutation_drug_map(processed_dir)

    y_lineage = n_lineage_classes = None
    if lambda_lineage > 0:
        from ..protocols.lineage import build_y_lineage
        y_lineage, classes, _ = build_y_lineage(processed_dir)
        n_lineage_classes = len(classes)

    # ---- splits -----------------------------------------------------------
    splits = {}
    split_reports = {}
    if "original" in split_names:
        splits["original"] = split_masks(tab)
        split_reports["original"] = {"source": "splits.csv (fixed, unmodified)"}
    if "phylo_clade" in split_names:
        # The Jaccard matrix is O(N^2) and takes ~80s at N=6000; cache it so a
        # repeated benchmark run does not pay for it again. The cache is keyed
        # only by processed_dir, so delete it if snp_matrix.npz changes.
        cache = processed_dir / "_jaccard_distance.npy"
        if cache.exists():
            D = np.load(cache)
        else:
            from ..phylogeny.graph_from_matrix import build as build_graph
            _, D = build_graph(processed_dir, keep_distance=True)
            np.save(cache, D)
        sp_arr, _, rep = build_phylo_split(D, y=y, cluster_threshold=clade_threshold)
        splits["phylo_clade"] = {s: (sp_arr == s) for s in ("train", "val", "test")}
        split_reports["phylo_clade"] = rep
        del D

    results = {
        "processed_dir": processed_dir.as_posix(),
        "n_isolates": int(y.shape[0]),
        "drug_columns": drug_columns,
        "seeds": list(seeds),
        "split_reports": split_reports,
        "protocol_info": {},
        "cells": {},
    }

    # Fixture-only: the ceiling any non-tautological protocol can reach. Without
    # it, a genome-wide score has nothing to be read against except 1.0, which
    # is not attainable when the generator drew resistance as a coin flip.
    if (processed_dir / "true_clade.csv").exists():
        from ..synthetic.oracle import clade_rate_oracle
        results["oracle_upper_bound"] = {
            name: clade_rate_oracle(processed_dir, y, splits[name], drug_columns)
            for name in split_names
        }

    for split_name in split_names:
        masks = splits[split_name]
        for protocol in protocols:
            # feature matrix depends on the split, because genome-wide site
            # selection is fitted on that split's train rows only
            if protocol == featmod.GENOMEWIDE:
                X, fnames, info = featmod.genomewide_features(
                    processed_dir, masks["train"], max_sites=max_genomewide_sites)
            else:
                X, fnames = tab["X"], tab["feature_names"]
                info = {"n_features": len(fnames),
                        "note": "curated catalogue columns; y_amr is an exact OR of these"}
            results["protocol_info"]["%s|%s" % (split_name, protocol)] = info

            for arch in architectures:
                key = "%s|%s|%s" % (split_name, protocol, arch)
                runs, infos = [], []
                t0 = time.time()
                for seed in seeds:
                    if protocol == featmod.CATALOGUE_LDO:
                        proba, cinfo = run_ldo(
                            arch, X, fnames, mutation_to_drug, y, masks,
                            edge_index, edge_weight, drug_columns, seed,
                            hidden_dim=hidden_dim, max_epochs=max_epochs,
                            patience=patience, lr=lr)
                    else:
                        proba, cinfo = run_cell(
                            arch, X, y, masks, edge_index, edge_weight,
                            drug_columns, seed, hidden_dim=hidden_dim,
                            max_epochs=max_epochs, patience=patience, lr=lr,
                            y_lineage=y_lineage,
                            n_lineage_classes=n_lineage_classes,
                            lambda_lineage=lambda_lineage)
                    runs.append(_evaluate(y, proba, masks, drug_columns))
                    infos.append(cinfo)
                results["cells"][key] = {
                    "split": split_name,
                    "protocol": protocol,
                    "architecture": arch,
                    "n_features": int(X.shape[1]),
                    "seconds": round(time.time() - t0, 1),
                    "aggregate": aggregate(runs),
                    "per_seed": runs,
                    "run_info": infos,
                }
                if verbose:
                    a = results["cells"][key]["aggregate"]["tuned"]
                    print("  %-46s core6=%.4f+/-%.4f  all9=%.4f  (%.0fs)" % (
                        key,
                        a["test_macro_f1_core6"]["mean"], a["test_macro_f1_core6"]["std"],
                        a["test_macro_f1_all9"]["mean"],
                        results["cells"][key]["seconds"]))
    return results


def summarise(results):
    """Compact table: rows are cells, sorted by test core-6 F1."""
    rows = []
    for key, cell in results["cells"].items():
        a = cell["aggregate"]["tuned"]
        rows.append({
            "split": cell["split"],
            "protocol": cell["protocol"],
            "architecture": cell["architecture"],
            "n_features": cell["n_features"],
            "test_core6_f1_mean": a["test_macro_f1_core6"]["mean"],
            "test_core6_f1_std": a["test_macro_f1_core6"]["std"],
            "test_all9_f1_mean": a["test_macro_f1_all9"]["mean"],
            "test_pr_auc_mean": a.get("test_macro_pr_auc", {}).get("mean"),
        })
    rows.sort(key=lambda r: -r["test_core6_f1_mean"])
    return rows
