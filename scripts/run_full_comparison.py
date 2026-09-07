"""
LAND-JEPA — Comprehensive Core Model Comparison & Label-Efficiency Experiment

Compares:
  Model 1: XGBoost Baseline
  Model 2: Supervised Causal TCN (Trained from Scratch)
  Model 3: JEPA-Pretrained TCN (Downstream Transfer)
  Model 4: Final Fused LAND-JEPA (Temporal + Static Terrain + InSAR + Physics)

Across label fractions: [1%, 5%, 10%, 25%, 50%, 100%]
Evaluates: Recall, FNR, Precision, F1, PR-AUC, Brier score, ECE, Inference Latency

Generates:
  - results/model_comparison.csv
  - results/label_efficiency.csv
  - results/label_efficiency_recall.png
  - results/label_efficiency_pr_auc.png
  - results/label_efficiency_f1.png
  - results/label_efficiency_fnr.png
  - results/model_comparison.png
  - results/calibration_curves.png
  - results/pr_curves.png
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.baselines.xgboost_trainer import XGBoostTrainer
from ml.evaluation.calibration import expected_calibration_error as compute_ece
from ml.evaluation.metrics import select_threshold_on_val
from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
from ml.features.physics_state import PhysicsStateEstimator
from ml.features.rainfall_features import compute_rainfall_features
from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
from ml.ingestion.demo.terrain_landslide_demo import (
    DemoLandslideInventoryProvider,
    DemoTerrainProvider,
)
from ml.ingestion.demo.weather_demo import DemoWeatherProvider
from ml.models.land_jepa_model import LandJEPARiskModel
from ml.models.registry import ModelRegistry
from ml.models.tcn_classifier import TCNClassifier, TCNFineTuneClassifier
from ml.models.tcn_encoder import TCNEncoder
from ml.preprocessing.normalizers import FeatureNormalizer, TemporalNormalizer
from ml.training.early_stopping import EarlyStopping
from ml.training.jepa_trainer import JEPATrainer
from ml.training.tcn_dataset import LandslideSequenceDataset, make_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEMO_ZONE_IDS = [
    "DEMO-NER-001", "DEMO-NER-002", "DEMO-NER-003", "DEMO-NER-004",
    "DEMO-NER-005", "DEMO-NER-006", "DEMO-NER-007", "DEMO-NER-008",
]


def load_all_data(seed: int = 42):
    """Load unified features across all zones for exact split alignment."""
    end = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=3 * 365)

    logger.info("Generating demo environmental data...")
    rain = DemoRainfallProvider({"seed": seed})
    rain_df = asyncio.run(rain.fetch(DEMO_ZONE_IDS, start, end))
    rain_df = rain.validate(rain_df)
    rain_df, _ = rain.transform(rain_df)
    rain_df = compute_rainfall_features(rain_df)

    wx = DemoWeatherProvider({"seed": seed})
    wx_df = asyncio.run(wx.fetch(DEMO_ZONE_IDS, start, end))
    wx_df = wx.validate(wx_df)
    wx_df, _ = wx.transform(wx_df)

    phys = PhysicsStateEstimator()
    phys_df = phys.compute(rain_df[["zone_id", "observed_at", "precipitation_mm"]])

    merged = rain_df.merge(
        wx_df[["zone_id", "observed_at", "temperature_c", "humidity_pct", "wind_speed_ms"]],
        on=["zone_id", "observed_at"], how="left"
    ).merge(
        phys_df[["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]],
        on=["zone_id", "observed_at"], how="left"
    )
    for drop in ["data_source", "is_demo", "quality_flag", "precipitation_mm"]:
        if drop in merged.columns:
            merged = merged.drop(columns=[drop])

    terrain_p = DemoTerrainProvider({"seed": seed})
    terrain_df = asyncio.run(terrain_p.fetch(DEMO_ZONE_IDS, start, end))
    terrain_df = terrain_p.validate(terrain_df)
    terrain_df, _ = terrain_p.transform(terrain_df)

    events_p = DemoLandslideInventoryProvider({"seed": seed, "n_events": 200})
    events_df = asyncio.run(events_p.fetch(DEMO_ZONE_IDS, start, end))
    events_df = events_p.validate(events_df)
    events_df, _ = events_p.transform(events_df)

    ds_cfg = DatasetConfig(
        context_hours=168,
        target_hours=24,
        stride_hours=24,
        min_valid_fraction=0.70,
        test_cutoff="2023-07-01",
        val_cutoff="2023-01-01",
    )
    builder = DatasetBuilder(ds_cfg)
    return builder, merged, terrain_df, events_df


def train_eval_xgboost(X_tr, y_tr, X_val, y_val, X_te, y_te, seed=42):
    """Model 1: XGBoost Baseline."""
    import xgboost as xgb
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    scaler = FeatureNormalizer(scaler_type="robust")
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )
    clf.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

    val_probs = clf.predict_proba(X_val_s)[:, 1]
    threshold = select_threshold_on_val(y_val, val_probs) if y_val.sum() > 0 else 0.35

    t0 = time.perf_counter()
    test_probs = clf.predict_proba(X_te_s)[:, 1]
    latency_ms = ((time.perf_counter() - t0) / max(len(X_te), 1)) * 1000.0

    return test_probs, threshold, latency_ms, clf


def train_eval_supervised_tcn(train_loader, val_loader, test_loader, input_dim, device, seed=42, epochs=5):
    """Model 2: Supervised TCN Baseline (trained from scratch)."""
    torch.manual_seed(seed)
    model = TCNClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        num_blocks=4,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    all_y = torch.cat([y for _, y in train_loader])
    pos_weight = torch.tensor([(len(all_y) - all_y.sum().item()) / max(all_y.sum().item(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    early_stopping = EarlyStopping(patience=5, mode="max")
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_probs, v_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                v_probs.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
                v_labels.extend(y.numpy())
        v_aucpr = average_precision_score(v_labels, v_probs) if sum(v_labels) > 0 else 0.0
        if early_stopping.step(v_aucpr, model):
            break

    early_stopping.restore(model)
    threshold = select_threshold_on_val(np.array(v_labels), np.array(v_probs)) if sum(v_labels) > 0 else 0.35

    model.eval()
    t0 = time.perf_counter()
    te_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            te_probs.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
    latency_ms = ((time.perf_counter() - t0) / max(len(te_probs), 1)) * 1000.0

    return np.array(te_probs, dtype=np.float32), threshold, latency_ms, model


def train_eval_jepa_tcn(train_loader, val_loader, test_loader, input_dim, checkpoint_dir, device, seed=42, epochs=5):
    """Model 3: JEPA-Pretrained TCN (fine-tuned)."""
    torch.manual_seed(seed)
    encoder = JEPATrainer.load_context_encoder(checkpoint_dir, input_dim=input_dim).to(device)
    model = TCNFineTuneClassifier(
        encoder=encoder,
        head_hidden_dim=64,
        dropout=0.1,
        freeze_encoder=False,
    ).to(device)

    all_y = torch.cat([y for _, y in train_loader])
    pos_weight = torch.tensor([(len(all_y) - all_y.sum().item()) / max(all_y.sum().item(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 0.0001},
        {"params": model.head.parameters(), "lr": 0.001},
    ], weight_decay=0.0001)

    early_stopping = EarlyStopping(patience=5, mode="max")
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_probs, v_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                v_probs.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
                v_labels.extend(y.numpy())
        v_aucpr = average_precision_score(v_labels, v_probs) if sum(v_labels) > 0 else 0.0
        if early_stopping.step(v_aucpr, model):
            break

    early_stopping.restore(model)
    threshold = select_threshold_on_val(np.array(v_labels), np.array(v_probs)) if sum(v_labels) > 0 else 0.35

    model.eval()
    t0 = time.perf_counter()
    te_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            te_probs.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
    latency_ms = ((time.perf_counter() - t0) / max(len(te_probs), 1)) * 1000.0

    return np.array(te_probs, dtype=np.float32), threshold, latency_ms, model


def train_eval_fused_land_jepa(
    train_seq, train_tab, y_tr,
    val_seq, val_tab, y_val,
    test_seq, test_tab, y_te,
    checkpoint_dir, device, seed=42,
    epochs=5,
):
    """Model 4: Full Fused LAND-JEPA (Temporal + Terrain + InSAR + Physics)."""
    torch.manual_seed(seed)
    model = LandJEPARiskModel(
        temporal_dim=train_seq.shape[-1],
        terrain_dim=8,
        insar_dim=3,
        physics_dim=3,
        tcn_hidden_dim=64,
        tcn_num_blocks=4,
        pretrained_encoder_path=checkpoint_dir,
    ).to(device)

    terr_tr = torch.from_numpy(train_tab[:, :8].astype(np.float32)).to(device)
    phys_tr = torch.from_numpy(train_tab[:, 8:11].astype(np.float32)).to(device)
    seq_tr = torch.from_numpy(train_seq.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1).to(device)

    terr_val = torch.from_numpy(val_tab[:, :8].astype(np.float32)).to(device)
    phys_val = torch.from_numpy(val_tab[:, 8:11].astype(np.float32)).to(device)
    seq_val = torch.from_numpy(val_seq.astype(np.float32)).to(device)

    terr_te = torch.from_numpy(test_tab[:, :8].astype(np.float32)).to(device)
    phys_te = torch.from_numpy(test_tab[:, 8:11].astype(np.float32)).to(device)
    seq_te = torch.from_numpy(test_seq.astype(np.float32)).to(device)

    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    early_stopping = EarlyStopping(patience=5, mode="max")

    batch_size = 64
    n_batches = int(np.ceil(len(seq_tr) / batch_size))
    n_val_batches = int(np.ceil(len(seq_val) / batch_size))

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(seq_tr))
        for b in range(n_batches):
            idx = indices[b * batch_size : (b + 1) * batch_size]
            optimizer.zero_grad()
            out = model(seq_tr[idx], terr_tr[idx], x_physics=phys_tr[idx])
            loss = criterion(out["logits_0h"], y_tr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_probs_list = []
        with torch.no_grad():
            for vb in range(n_val_batches):
                v_idx = slice(vb * batch_size, (vb + 1) * batch_size)
                v_out = model(seq_val[v_idx], terr_val[v_idx], x_physics=phys_val[v_idx])
                v_probs_list.append(torch.sigmoid(v_out["logits_0h"]).squeeze(1).cpu().numpy())
            v_probs = np.concatenate(v_probs_list) if v_probs_list else np.zeros(len(y_val))

        v_aucpr = average_precision_score(y_val, v_probs) if y_val.sum() > 0 else 0.0
        if early_stopping.step(v_aucpr, model):
            break

    early_stopping.restore(model)
    threshold = select_threshold_on_val(y_val, v_probs) if y_val.sum() > 0 else 0.35

    model.eval()
    t0 = time.perf_counter()
    n_te_batches = int(np.ceil(len(seq_te) / batch_size))
    te_probs_list = []
    with torch.no_grad():
        for tb in range(n_te_batches):
            t_idx = slice(tb * batch_size, (tb + 1) * batch_size)
            t_out = model(seq_te[t_idx], terr_te[t_idx], x_physics=phys_te[t_idx])
            te_probs_list.append(torch.sigmoid(t_out["logits_0h"]).squeeze(1).cpu().numpy())
    te_probs = np.concatenate(te_probs_list) if te_probs_list else np.zeros(len(y_te))
    latency_ms = ((time.perf_counter() - t0) / max(len(te_probs), 1)) * 1000.0

    return te_probs, threshold, latency_ms, model


def compute_metrics_dict(y_true, y_prob, threshold, latency_ms):
    """Compute all evaluation metrics accurately."""
    y_pred = (y_prob >= threshold).astype(int)
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    fnr = float(1.0 - rec)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    aucpr = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_true, y_prob, n_bins=10))

    return {
        "recall": round(rec, 4),
        "fnr": round(fnr, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "aucpr": round(aucpr, 4),
        "auroc": round(auroc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "latency_ms": round(latency_ms, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Full LAND-JEPA Core Model Comparison & Label-Efficiency Experiment")
    parser.add_argument("--jepa-checkpoint", default="ml/checkpoints/jepa_pretrained")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Run fast test mode")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    logger.info("LAND-JEPA — Controlled 4-Model Comparison & Label-Efficiency Sweep")
    logger.info(f"Device: {device} | Seed: {args.seed} | Fast Mode: {args.fast}")
    logger.info("=" * 80)

    builder, merged, terrain_df, events_df = load_all_data(args.seed)

    # 1. Full Dataset for Baseline Normalization
    train_full, val_full, test_full = builder.build(
        merged_df=merged, terrain_df=terrain_df, events_df=events_df,
        label_fraction=1.0, label_seed=args.seed,
    )

    temporal_norm = TemporalNormalizer(scaler_type="robust")
    temporal_norm.fit(train_full.X_sequence)
    val_seq_norm = temporal_norm.transform(val_full.X_sequence)
    test_seq_norm = temporal_norm.transform(test_full.X_sequence)

    tabular_norm = FeatureNormalizer(scaler_type="robust")
    X_tr_tab_norm = tabular_norm.fit_transform(train_full.X_tabular)
    X_val_tab_norm = tabular_norm.transform(val_full.X_tabular)
    X_te_tab_norm = tabular_norm.transform(test_full.X_tabular)

    train_full_seq_norm = temporal_norm.transform(train_full.X_sequence)
    val_seq_ds = LandslideSequenceDataset(val_seq_norm, val_full.y, augment=False)
    test_seq_ds = LandslideSequenceDataset(test_seq_norm, test_full.y, augment=False)
    _, val_loader = make_dataloaders(val_seq_ds, val_seq_ds, batch_size=64)
    _, test_loader = make_dataloaders(test_seq_ds, test_seq_ds, batch_size=64)

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0] if not args.fast else [0.05, 0.25, 1.0]
    models = ["XGBoost", "Supervised TCN", "JEPA-TCN", "Fused LAND-JEPA"]

    records = []
    prob_curves = {}  # for calibration and PR plots at 100% fraction

    for frac in fractions:
        logger.info(f"\n========================================================")
        logger.info(f"Evaluating Label Fraction: {frac:.0%}")
        logger.info(f"========================================================")

        if frac < 1.0:
            pos_indices = np.where(train_full.y == 1)[0]
            n_keep = max(1, int(round(len(pos_indices) * frac)))
            rng = np.random.default_rng(args.seed)
            keep_pos = set(rng.choice(pos_indices, size=n_keep, replace=False))
            y_sub = np.zeros_like(train_full.y)
            for idx in keep_pos:
                y_sub[idx] = 1.0
            epochs_to_train = 5
        else:
            y_sub = train_full.y.copy()
            epochs_to_train = 12

        train_seq_ds = LandslideSequenceDataset(train_full_seq_norm, y_sub, augment=True)
        train_loader, _ = make_dataloaders(train_seq_ds, val_seq_ds, batch_size=64)

        # ── MODEL 1: XGBoost ──────────────────────────────────────────
        logger.info(f"Training XGBoost (frac={frac:.0%})...")
        p1, t1, lat1, _ = train_eval_xgboost(
            X_tr_tab_norm, y_sub, X_val_tab_norm, val_full.y, X_te_tab_norm, test_full.y, seed=args.seed
        )
        m1 = compute_metrics_dict(test_full.y, p1, t1, lat1)
        records.append({"model": "XGBoost", "label_fraction": frac, **m1})
        if frac == 1.0: prob_curves["XGBoost"] = p1

        # ── MODEL 2: Supervised TCN ───────────────────────────────────
        logger.info(f"Training Supervised TCN (frac={frac:.0%}, epochs={epochs_to_train})...")
        p2, t2, lat2, _ = train_eval_supervised_tcn(
            train_loader, val_loader, test_loader, train_full.X_sequence.shape[-1], device, seed=args.seed, epochs=epochs_to_train
        )
        m2 = compute_metrics_dict(test_full.y, p2, t2, lat2)
        records.append({"model": "Supervised TCN", "label_fraction": frac, **m2})
        if frac == 1.0: prob_curves["Supervised TCN"] = p2

        # ── MODEL 3: JEPA-TCN ─────────────────────────────────────────
        logger.info(f"Training JEPA-TCN (frac={frac:.0%}, epochs={epochs_to_train})...")
        p3, t3, lat3, _ = train_eval_jepa_tcn(
            train_loader, val_loader, test_loader, train_full.X_sequence.shape[-1], args.jepa_checkpoint, device, seed=args.seed, epochs=epochs_to_train
        )
        m3 = compute_metrics_dict(test_full.y, p3, t3, lat3)
        records.append({"model": "JEPA-TCN", "label_fraction": frac, **m3})
        if frac == 1.0: prob_curves["JEPA-TCN"] = p3

        # ── MODEL 4: Fused LAND-JEPA ──────────────────────────────────
        logger.info(f"Training Fused LAND-JEPA (frac={frac:.0%}, epochs={epochs_to_train})...")
        p4, t4, lat4, final_land_jepa_model = train_eval_fused_land_jepa(
            train_full_seq_norm, X_tr_tab_norm, y_sub,
            val_seq_norm, X_val_tab_norm, val_full.y,
            test_seq_norm, X_te_tab_norm, test_full.y,
            args.jepa_checkpoint, device, seed=args.seed,
            epochs=epochs_to_train,
        )
        m4 = compute_metrics_dict(test_full.y, p4, t4, lat4)
        records.append({"model": "Fused LAND-JEPA", "label_fraction": frac, **m4})
        if frac == 1.0:
            prob_curves["Fused LAND-JEPA"] = p4
            # Save the production LandJEPARiskModel
            ckpt_path = Path("ml/checkpoints/land_jepa_production")
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(final_land_jepa_model.state_dict(), ckpt_path / "land_jepa_weights.pt")

            # Register into ModelRegistry
            registry = ModelRegistry()
            registry.register(
                model_name="LAND-JEPA",
                version="v1.0.0",
                checkpoint_path=str(ckpt_path / "land_jepa_weights.pt"),
                metrics=m4,
                configuration={"temporal_encoder": "JEPA-TCN", "fusion": "gated", "horizon": [0, 24, 48]},
                status="PRODUCTION",
                notes="Full Fused Multimodal LAND-JEPA Model",
            )
            registry.promote_to_production("LAND-JEPA", "v1.0.0")

    df_results = pd.DataFrame(records)
    df_results.to_csv(results_path / "label_efficiency.csv", index=False)
    logger.info(f"Saved: {results_path / 'label_efficiency.csv'}")

    # Model comparison table (100% labels)
    df_comp = df_results[df_results["label_fraction"] == 1.0].copy()
    df_comp.to_csv(results_path / "model_comparison.csv", index=False)
    logger.info(f"Saved: {results_path / 'model_comparison.csv'}")

    # ── GENERATE 7 PUBLICATION-QUALITY PLOTS ──────────────────────────
    logger.info("Generating publication-quality comparison figures...")
    colors = {
        "XGBoost": "#f59e0b",
        "Supervised TCN": "#8b5cf6",
        "JEPA-TCN": "#06b6d4",
        "Fused LAND-JEPA": "#10b981",
    }

    # 1. Label Efficiency: Recall
    plt.figure(figsize=(8, 5))
    for m in models:
        sub = df_results[df_results["model"] == m]
        plt.plot(sub["label_fraction"] * 100, sub["recall"], marker="o", label=m, color=colors[m], linewidth=2)
    plt.title("Label Efficiency: Recall vs. Labeled Data Fraction", fontsize=14, fontweight="bold")
    plt.xlabel("Labeled Data Fraction (%)", fontsize=12)
    plt.ylabel("Test Recall ↑", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "label_efficiency_recall.png", dpi=300)
    plt.close()

    # 2. Label Efficiency: PR-AUC
    plt.figure(figsize=(8, 5))
    for m in models:
        sub = df_results[df_results["model"] == m]
        plt.plot(sub["label_fraction"] * 100, sub["aucpr"], marker="s", label=m, color=colors[m], linewidth=2)
    plt.title("Label Efficiency: PR-AUC vs. Labeled Data Fraction", fontsize=14, fontweight="bold")
    plt.xlabel("Labeled Data Fraction (%)", fontsize=12)
    plt.ylabel("Test PR-AUC ↑", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "label_efficiency_pr_auc.png", dpi=300)
    plt.close()

    # 3. Label Efficiency: F1 Score
    plt.figure(figsize=(8, 5))
    for m in models:
        sub = df_results[df_results["model"] == m]
        plt.plot(sub["label_fraction"] * 100, sub["f1"], marker="^", label=m, color=colors[m], linewidth=2)
    plt.title("Label Efficiency: F1 Score vs. Labeled Data Fraction", fontsize=14, fontweight="bold")
    plt.xlabel("Labeled Data Fraction (%)", fontsize=12)
    plt.ylabel("Test F1 Score ↑", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "label_efficiency_f1.png", dpi=300)
    plt.close()

    # 4. Label Efficiency: False Negative Rate (FNR)
    plt.figure(figsize=(8, 5))
    for m in models:
        sub = df_results[df_results["model"] == m]
        plt.plot(sub["label_fraction"] * 100, sub["fnr"], marker="d", label=m, color=colors[m], linewidth=2)
    plt.title("Label Efficiency: False Negative Rate (FNR) vs. Labeled Fraction", fontsize=14, fontweight="bold")
    plt.xlabel("Labeled Data Fraction (%)", fontsize=12)
    plt.ylabel("Test FNR (Missed Landslides) ↓", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "label_efficiency_fnr.png", dpi=300)
    plt.close()

    # 5. Core Model Comparison Bar Chart (100% fraction)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.20
    plt.bar(x - 1.5 * width, df_comp["recall"], width, label="Recall", color="#3b82f6")
    plt.bar(x - 0.5 * width, df_comp["f1"], width, label="F1", color="#10b981")
    plt.bar(x + 0.5 * width, df_comp["aucpr"], width, label="PR-AUC", color="#8b5cf6")
    plt.bar(x + 1.5 * width, df_comp["auroc"], width, label="AUROC", color="#f59e0b")
    plt.xticks(x, models, fontsize=11, fontweight="bold")
    plt.title("Core Model Performance Comparison (Full Dataset)", fontsize=14, fontweight="bold")
    plt.ylabel("Score [0, 1]", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(results_path / "model_comparison.png", dpi=300)
    plt.close()

    # 6. Calibration Curves
    plt.figure(figsize=(8, 6))
    for m in models:
        if m in prob_curves:
            prob_true, prob_pred = calibration_curve(test_full.y, prob_curves[m], n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=f"{m} (ECE={df_comp[df_comp['model']==m]['ece'].values[0]:.3f})", color=colors[m], linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    plt.title("Reliability Calibration Curves (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positive Landslides", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "calibration_curves.png", dpi=300)
    plt.close()

    # 7. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for m in models:
        if m in prob_curves:
            p, r, _ = precision_recall_curve(test_full.y, prob_curves[m])
            auc = df_comp[df_comp['model']==m]['aucpr'].values[0]
            plt.plot(r, p, label=f"{m} (PR-AUC={auc:.3f})", color=colors[m], linewidth=2)
    plt.title("Precision-Recall Curves (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall ↑", fontsize=12)
    plt.ylabel("Precision ↑", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(results_path / "pr_curves.png", dpi=300)
    plt.close()

    logger.info("All 7 plots successfully generated and saved to results/")

    # Print summary table
    print("\n" + "=" * 95)
    print("FINAL 4-WAY MODEL COMPARISON (100% LABELED DATA)")
    print("=" * 95)
    header = f"{'Model':<18} {'Recall':>8} {'FNR':>8} {'Precision':>10} {'F1':>8} {'PR-AUC':>9} {'AUROC':>8} {'Brier':>8} {'ECE':>7} {'Latency':>10}"
    print(header)
    print("-" * 95)
    for _, row in df_comp.iterrows():
        print(
            f"{row['model']:<18} "
            f"{row['recall']:>8.4f} "
            f"{row['fnr']:>8.4f} "
            f"{row['precision']:>10.4f} "
            f"{row['f1']:>8.4f} "
            f"{row['aucpr']:>9.4f} "
            f"{row['auroc']:>8.4f} "
            f"{row['brier_score']:>8.4f} "
            f"{row['ece']:>7.4f} "
            f"{row['latency_ms']:>8.3f}ms"
        )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
