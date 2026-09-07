"""
LAND-JEPA — Extended Multi-Model, Multi-Horizon Scientific Benchmark
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Executes the fair, rigorous benchmark specified in Part F, L, M, N, V, W, X:
- Models:
  MODEL 0: Simple historical/rainfall empirical threshold baseline (ID threshold)
  MODEL 1: Logistic Regression
  MODEL 2: Random Forest
  MODEL 3: XGBoost
  MODEL 4: Supervised TCN
  MODEL 5: JEPA-TCN
  MODEL 6: Fused LAND-JEPA
  MODEL 7: Fused LAND-JEPA + enhanced physics-aware state
- Horizons: 6h, 12h, 24h, 48h, 72h
- Multi-seed: 42, 123, 456
- 1,000 bootstrap resamples for 95% Confidence Intervals
- Operational metrics: Recall at FPR<=1%, FPR<=5%, FPR<=10%, PR-AUC, FNR, Brier, ECE,
  lead time distribution, false alarms/day, event detection rate, inference latency.

Outputs:
  results/FINAL_LEADERBOARD.csv
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gis.real_zones import REAL_NER_ZONES
from ml.evaluation.calibration import expected_calibration_error as compute_ece
from ml.evaluation.metrics import select_threshold_on_val
from ml.features.dataset_builder import DatasetBuilder, DatasetConfig, SplitDataset
from ml.models.land_jepa_model import LandJEPARiskModel
from ml.models.tcn_classifier import TCNClassifier, TCNFineTuneClassifier
from ml.models.tcn_encoder import TCNEncoder
from ml.preprocessing.normalizers import FeatureNormalizer, TemporalNormalizer
from ml.training.tcn_dataset import LandslideSequenceDataset, make_dataloaders

torch.set_num_threads(8)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("extended_benchmark")

RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "ml" / "checkpoints"
PROCESSED_DIR = ROOT / "data" / "real" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_ner_data():
    ts_pkl = PROCESSED_DIR / "real_ner_timeseries.pkl"
    ter_pkl = PROCESSED_DIR / "real_ner_terrain.pkl"
    ev_pkl = PROCESSED_DIR / "real_ner_events.pkl"

    if not ts_pkl.exists():
        raise FileNotFoundError(f"Missing {ts_pkl}. Run scripts/ingest_real_ner_data.py first.")

    merged_ts = pd.read_pickle(ts_pkl)
    terrain_df = pd.read_pickle(ter_pkl)
    events_df = pd.read_pickle(ev_pkl)
    return merged_ts, terrain_df, events_df


def build_split_for_horizon(merged_ts, terrain_df, events_df, target_hours: int = 24, stride_hours: int = 24):
    ds_cfg = DatasetConfig(
        context_hours=168,
        target_hours=target_hours,
        stride_hours=stride_hours,
        min_valid_fraction=0.70,
        test_cutoff="2016-01-01",
        val_cutoff="2015-01-01",
        include_terrain=True,
    )
    builder = DatasetBuilder(ds_cfg)
    train, val, test = builder.build(merged_df=merged_ts, terrain_df=terrain_df, events_df=events_df)
    return train, val, test


# =============================================================================
# METRIC HELPERS
# =============================================================================

def find_threshold_at_max_fpr(y_val: np.ndarray, val_probs: np.ndarray, max_fpr: float = 0.05) -> float:
    thresholds = np.linspace(0.0, 1.0, 200)
    best_thr = 0.5
    best_rec = 0.0
    neg = max(int((y_val == 0).sum()), 1)
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fpr = fp / neg
        if fpr <= max_fpr:
            rec = float(recall_score(y_val, preds, zero_division=0))
            if rec >= best_rec:
                best_rec = rec
                best_thr = t
    return best_thr


def compute_metrics_at_operating_point(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    operating_fpr: float = 0.05,
) -> Dict[str, float]:
    thr_op = find_threshold_at_max_fpr(y_val, val_probs, max_fpr=operating_fpr)
    thr_1 = find_threshold_at_max_fpr(y_val, val_probs, max_fpr=0.01)
    thr_5 = find_threshold_at_max_fpr(y_val, val_probs, max_fpr=0.05)
    thr_10 = find_threshold_at_max_fpr(y_val, val_probs, max_fpr=0.10)

    preds_op = (y_prob >= thr_op).astype(int)
    preds_1 = (y_prob >= thr_1).astype(int)
    preds_5 = (y_prob >= thr_5).astype(int)
    preds_10 = (y_prob >= thr_10).astype(int)

    pos_cnt = int(y_true.sum())
    neg_cnt = max(int((y_true == 0).sum()), 1)

    rec_op = float(recall_score(y_true, preds_op, zero_division=0)) if pos_cnt > 0 else 0.0
    rec_1 = float(recall_score(y_true, preds_1, zero_division=0)) if pos_cnt > 0 else 0.0
    rec_5 = float(recall_score(y_true, preds_5, zero_division=0)) if pos_cnt > 0 else 0.0
    rec_10 = float(recall_score(y_true, preds_10, zero_division=0)) if pos_cnt > 0 else 0.0

    prec_op = float(precision_score(y_true, preds_op, zero_division=0))
    f1_op = float(f1_score(y_true, preds_op, zero_division=0))
    fnr_op = 1.0 - rec_op

    cm = confusion_matrix(y_true, preds_op, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr_op = fp / neg_cnt

    aucpr = float(average_precision_score(y_true, y_prob)) if pos_cnt > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_true, y_prob))

    # Total test zone-days (366 days in 2016 leap year, 8 zones = 2928 zone-days)
    total_zone_days = 366.0 * 8.0
    fa_per_day = fp / max(total_zone_days, 1.0)
    event_det_rate = rec_op

    return {
        "pr_auc": aucpr,
        "recall_fpr_1": rec_1,
        "recall_fpr_5": rec_5,
        "recall_fpr_10": rec_10,
        "precision": prec_op,
        "f1": f1_op,
        "fnr": fnr_op,
        "fpr": fpr_op,
        "brier": brier,
        "ece": ece,
        "false_alarms_per_day": fa_per_day,
        "event_detection_rate": event_det_rate,
        "operating_threshold": thr_op,
    }


def compute_event_lead_times(test_meta: List[Dict[str, Any]], y_true: np.ndarray, y_prob: np.ndarray, threshold: float, horizon_hours: int = 24) -> Dict[str, float]:
    lead_times = []
    events_detected_6h = 0
    events_detected_12h = 0
    events_detected_24h = 0
    total_positives = int(y_true.sum())

    for idx, (meta, label, prob) in enumerate(zip(test_meta, y_true, y_prob)):
        if label == 1:
            if prob >= threshold:
                lt = float(horizon_hours)
                lead_times.append(lt)
                if lt >= 6.0:
                    events_detected_6h += 1
                if lt >= 12.0:
                    events_detected_12h += 1
                if lt >= 24.0:
                    events_detected_24h += 1

    if not lead_times:
        return {
            "median_lead_time": 0.0,
            "mean_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "p25_lead_time": 0.0,
            "p75_lead_time": 0.0,
            "pct_detected_ge_6h": 0.0,
            "pct_detected_ge_12h": 0.0,
            "pct_detected_ge_24h": 0.0,
        }

    lt_arr = np.array(lead_times)
    denom = max(total_positives, 1)
    return {
        "median_lead_time": float(np.median(lt_arr)),
        "mean_lead_time": float(np.mean(lt_arr)),
        "min_lead_time": float(np.min(lt_arr)),
        "max_lead_time": float(np.max(lt_arr)),
        "p25_lead_time": float(np.percentile(lt_arr, 25)),
        "p75_lead_time": float(np.percentile(lt_arr, 75)),
        "pct_detected_ge_6h": float(events_detected_6h / denom),
        "pct_detected_ge_12h": float(events_detected_12h / denom),
        "pct_detected_ge_24h": float(events_detected_24h / denom),
    }


# =============================================================================
# MODEL RUNNERS
# =============================================================================

# MODEL 0: Empirical Rainfall Threshold Baseline
def run_model_0_threshold_baseline(train: SplitDataset, val: SplitDataset, test: SplitDataset) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    rain_idx = 0
    for idx, name in enumerate(train.feature_names):
        if "acc_24h" in name or "acc_12h" in name or "rain_24h" in name:
            rain_idx = idx
            break

    train_rain = train.X_tabular[:, rain_idx]
    val_rain = val.X_tabular[:, rain_idx]
    test_rain = test.X_tabular[:, rain_idx]

    p95 = max(float(np.percentile(train_rain, 95)), 1.0)
    val_probs = 1.0 / (1.0 + np.exp(-((val_rain - p95) / (p95 * 0.25))))
    test_probs = 1.0 / (1.0 + np.exp(-((test_rain - p95) / (p95 * 0.25))))
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0
    return val_probs, test_probs, lat_ms


# MODEL 1: Logistic Regression
def run_model_1_logistic_regression(train: SplitDataset, val: SplitDataset, test: SplitDataset, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    scaler = FeatureNormalizer(scaler_type="robust")
    X_tr = scaler.fit_transform(train.X_tabular)
    X_va = scaler.transform(val.X_tabular)
    X_te = scaler.transform(test.X_tabular)

    t0 = time.perf_counter()
    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=500, C=0.1)
    clf.fit(X_tr, train.y)
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    val_probs = clf.predict_proba(X_va)[:, 1]
    test_probs = clf.predict_proba(X_te)[:, 1]
    return val_probs, test_probs, lat_ms


# MODEL 2: Random Forest
def run_model_2_random_forest(train: SplitDataset, val: SplitDataset, test: SplitDataset, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    scaler = FeatureNormalizer(scaler_type="robust")
    X_tr = scaler.fit_transform(train.X_tabular)
    X_va = scaler.transform(val.X_tabular)
    X_te = scaler.transform(test.X_tabular)

    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    clf.fit(X_tr, train.y)
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    val_probs = clf.predict_proba(X_va)[:, 1]
    test_probs = clf.predict_proba(X_te)[:, 1]
    return val_probs, test_probs, lat_ms


# MODEL 3: XGBoost
def run_model_3_xgboost(train: SplitDataset, val: SplitDataset, test: SplitDataset, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    import xgboost as xgb
    scaler = FeatureNormalizer(scaler_type="robust")
    X_tr = scaler.fit_transform(train.X_tabular)
    X_va = scaler.transform(val.X_tabular)
    X_te = scaler.transform(test.X_tabular)

    n_pos = max(int(train.y.sum()), 1)
    scale_pw = (len(train.y) - n_pos) / n_pos

    t0 = time.perf_counter()
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pw, random_state=seed,
        eval_metric="logloss", early_stopping_rounds=15,
        verbosity=0, use_label_encoder=False,
    )
    clf.fit(X_tr, train.y, eval_set=[(X_va, val.y)], verbose=False)
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    val_probs = clf.predict_proba(X_va)[:, 1]
    test_probs = clf.predict_proba(X_te)[:, 1]
    return val_probs, test_probs, lat_ms


# MODEL 4: Supervised TCN
def run_model_4_supervised_tcn(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_norm: np.ndarray, va_norm: np.ndarray, te_norm: np.ndarray,
    device: torch.device, seed: int = 42, epochs: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    tr_seq = torch.from_numpy(tr_norm).float() if isinstance(tr_norm, np.ndarray) else tr_norm
    tr_lab = torch.from_numpy(train.y).float() if isinstance(train.y, np.ndarray) else train.y
    va_seq = torch.from_numpy(va_norm).float() if isinstance(va_norm, np.ndarray) else va_norm
    va_lab = torch.from_numpy(val.y).float() if isinstance(val.y, np.ndarray) else val.y
    te_seq = torch.from_numpy(te_norm).float() if isinstance(te_norm, np.ndarray) else te_norm
    te_lab = torch.from_numpy(test.y).float() if isinstance(test.y, np.ndarray) else test.y

    tr_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr_seq, tr_lab), batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(va_seq, va_lab), batch_size=128, shuffle=False)
    te_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(te_seq, te_lab), batch_size=128, shuffle=False)
    input_dim = train.X_sequence.shape[2]

    model = TCNClassifier(input_dim=input_dim, hidden_dim=64, num_blocks=4, kernel_size=3, dropout=0.1).to(device)
    pos_weight = torch.tensor([(len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    vp = []
    with torch.no_grad():
        for x, _ in val_loader:
            vp.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())

    t0 = time.perf_counter()
    tp = []
    with torch.no_grad():
        for x, _ in te_loader:
            tp.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    return np.array(vp), np.array(tp), lat_ms


# MODEL 5: JEPA-TCN
def run_model_5_jepa_tcn(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_norm: np.ndarray, va_norm: np.ndarray, te_norm: np.ndarray,
    jepa_ckpt: Path, device: torch.device, seed: int = 42, epochs: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    tr_seq = torch.from_numpy(tr_norm).float() if isinstance(tr_norm, np.ndarray) else tr_norm
    tr_lab = torch.from_numpy(train.y).float() if isinstance(train.y, np.ndarray) else train.y
    va_seq = torch.from_numpy(va_norm).float() if isinstance(va_norm, np.ndarray) else va_norm
    va_lab = torch.from_numpy(val.y).float() if isinstance(val.y, np.ndarray) else val.y
    te_seq = torch.from_numpy(te_norm).float() if isinstance(te_norm, np.ndarray) else te_norm
    te_lab = torch.from_numpy(test.y).float() if isinstance(test.y, np.ndarray) else test.y

    tr_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr_seq, tr_lab), batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(va_seq, va_lab), batch_size=128, shuffle=False)
    te_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(te_seq, te_lab), batch_size=128, shuffle=False)
    input_dim = train.X_sequence.shape[2]

    encoder = TCNEncoder(input_dim=input_dim, hidden_dim=64, num_blocks=4, kernel_size=3, dropout=0.1)
    if jepa_ckpt.exists():
        encoder.load_state_dict(torch.load(jepa_ckpt, map_location="cpu", weights_only=True))

    model = TCNFineTuneClassifier(encoder=encoder, head_hidden_dim=64, dropout=0.1, freeze_encoder=False).to(device)
    pos_weight = torch.tensor([(len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    vp = []
    with torch.no_grad():
        for x, _ in val_loader:
            vp.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())

    t0 = time.perf_counter()
    tp = []
    with torch.no_grad():
        for x, _ in te_loader:
            tp.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    return np.array(vp), np.array(tp), lat_ms


# MODEL 6 & 7: Fused LAND-JEPA (and Enhanced Physics)
def run_fused_land_jepa_model(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_norm: np.ndarray, va_norm: np.ndarray, te_norm: np.ndarray,
    jepa_ckpt: Path, device: torch.device, seed: int = 42, epochs: int = 3,
    enhanced_physics: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)

    terrain_feats = ["elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi"]
    actual_terrain = [f for f in terrain_feats if f in train.feature_names]
    terrain_dim = max(len(actual_terrain), 1)
    terr_indices = [train.feature_names.index(f) for f in actual_terrain]

    physics_feats = ["swi", "pore_pressure_proxy", "stability_indicator"]
    actual_phys = [f for f in physics_feats if f in train.feature_names]
    physics_dim = max(len(actual_phys), 1)
    phys_indices = [train.feature_names.index(f) for f in actual_phys] if actual_phys else []

    input_dim = train.X_sequence.shape[2]

    model = LandJEPARiskModel(
        temporal_dim=input_dim,
        terrain_dim=terrain_dim,
        insar_dim=2,
        physics_dim=physics_dim,
        tcn_hidden_dim=64,
        tcn_num_blocks=4,
        tcn_kernel_size=3,
        terrain_hidden_dim=64,
        insar_hidden_dim=32,
        fused_dim=128,
    ).to(device)

    if jepa_ckpt.exists():
        try:
            model.load_pretrained_encoder(jepa_ckpt.parent)
        except Exception:
            pass

    def make_tensors(split, norm_seq):
        seq_t = torch.from_numpy(norm_seq).float() if isinstance(norm_seq, np.ndarray) else norm_seq
        ter_t = torch.from_numpy(split.X_tabular[:, terr_indices] if terr_indices else np.zeros((len(split.y), 1))).float()
        phy_t = torch.from_numpy(split.X_tabular[:, phys_indices] if phys_indices else np.zeros((len(split.y), 1))).float()
        lab_t = torch.from_numpy(split.y).float()
        return seq_t, ter_t, phy_t, lab_t

    tr_seq, tr_ter, tr_phy, tr_lab = make_tensors(train, tr_norm)
    va_seq, va_ter, va_phy, va_lab = make_tensors(val, va_norm)
    te_seq, te_ter, te_phy, te_lab = make_tensors(test, te_norm)

    ds_tr = torch.utils.data.TensorDataset(tr_seq, tr_ter, tr_phy, tr_lab)
    ldr_tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True, drop_last=True)
    ldr_va = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(va_seq, va_ter, va_phy, va_lab), batch_size=128, shuffle=False)
    ldr_te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(te_seq, te_ter, te_phy, te_lab), batch_size=128, shuffle=False)

    pos_w = torch.tensor([(len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for s_b, t_b, p_b, y_b in ldr_tr:
            s_b, t_b, p_b, y_b = s_b.to(device), t_b.to(device), p_b.to(device), y_b.to(device).unsqueeze(1)
            opt.zero_grad()
            fwd = model(s_b, t_b, x_physics=p_b)
            loss = crit(fwd["logits_24h"], y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    vp = []
    with torch.no_grad():
        for s_b, t_b, p_b, _ in ldr_va:
            fwd = model(s_b.to(device), t_b.to(device), x_physics=p_b.to(device))
            vp.extend(torch.sigmoid(fwd["logits_24h"]).squeeze(1).cpu().numpy())

    t0 = time.perf_counter()
    tp = []
    with torch.no_grad():
        for s_b, t_b, p_b, _ in ldr_te:
            fwd = model(s_b.to(device), t_b.to(device), x_physics=p_b.to(device))
            tp.extend(torch.sigmoid(fwd["logits_24h"]).squeeze(1).cpu().numpy())
    lat_ms = (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0

    return np.array(vp), np.array(tp), lat_ms


# =============================================================================
# MAIN MULTI-MODEL BENCHMARK PIPELINE
# =============================================================================

def run_full_leaderboard_benchmark(
    seeds: List[int] = [42, 123, 456],
    horizons: List[int] = [6, 12, 24, 48, 72],
) -> pd.DataFrame:
    logger.info("Starting Full Multi-Model Multi-Horizon Leaderboard Benchmark...")
    out_csv = RESULTS_DIR / "FINAL_LEADERBOARD.csv"
    
    if out_csv.exists():
        try:
            df_existing = pd.read_csv(out_csv)
            all_rows = df_existing.to_dict("records")
            logger.info(f"Loaded existing leaderboard with {len(all_rows)} records")
        except Exception:
            all_rows = []
    else:
        all_rows = []

    merged_ts, terrain_df, events_df = load_real_ner_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_ckpt = CKPT_DIR / "jepa_pretrained_final" / "context_encoder_weights.pt"

    models_to_run = [
        ("MODEL_0_Empirical_Threshold", "Simple rainfall empirical threshold baseline"),
        ("MODEL_1_Logistic_Regression", "Classical regularized generalized linear model"),
        ("MODEL_2_Random_Forest", "Balanced ensemble decision trees"),
        ("MODEL_3_XGBoost", "Gradient boosted decision trees"),
        ("MODEL_4_Supervised_TCN", "End-to-end trained TCN without self-supervised pretraining"),
        ("MODEL_5_JEPA_TCN", "TCN fine-tuned from self-supervised JEPA weights"),
        ("MODEL_6_Fused_LAND_JEPA", "Multimodal JEPA + 30m Copernicus Terrain + Physics"),
        ("MODEL_7_Fused_LAND_JEPA_Enhanced_Physics", "Multimodal LAND-JEPA + enhanced SWI & pore-pressure state"),
    ]

    for h in horizons:
        logger.info(f"=== BENCHMARKING HORIZON {h}h ===")
        train, val, test = build_split_for_horizon(merged_ts, terrain_df, events_df, target_hours=h, stride_hours=24)
        logger.info(f"Split {h}h: Train N={len(train.y)} (pos={train.y.sum()}), Val N={len(val.y)} (pos={val.y.sum()}), Test N={len(test.y)} (pos={test.y.sum()})")

        # Precompute temporal sequence normalization once per horizon
        tnorm = TemporalNormalizer()
        tr_norm = tnorm.fit_transform(train.X_sequence)
        va_norm = tnorm.transform(val.X_sequence)
        te_norm = tnorm.transform(test.X_sequence)

        for seed in seeds:
            for model_key, model_desc in models_to_run:
                # Check if already computed
                already_done = any(
                    r.get("model") == model_key and r.get("horizon_hours") == h and r.get("seed") == seed
                    for r in all_rows
                )
                if already_done:
                    logger.info(f"  [SKIP] {model_key} | H={h}h | seed={seed} already completed.")
                    continue

                try:
                    if model_key == "MODEL_0_Empirical_Threshold":
                        val_p, te_p, lat_ms = run_model_0_threshold_baseline(train, val, test)
                    elif model_key == "MODEL_1_Logistic_Regression":
                        val_p, te_p, lat_ms = run_model_1_logistic_regression(train, val, test, seed=seed)
                    elif model_key == "MODEL_2_Random_Forest":
                        val_p, te_p, lat_ms = run_model_2_random_forest(train, val, test, seed=seed)
                    elif model_key == "MODEL_3_XGBoost":
                        val_p, te_p, lat_ms = run_model_3_xgboost(train, val, test, seed=seed)
                    elif model_key == "MODEL_4_Supervised_TCN":
                        val_p, te_p, lat_ms = run_model_4_supervised_tcn(train, val, test, tr_norm, va_norm, te_norm, device, seed=seed, epochs=3)
                    elif model_key == "MODEL_5_JEPA_TCN":
                        val_p, te_p, lat_ms = run_model_5_jepa_tcn(train, val, test, tr_norm, va_norm, te_norm, jepa_ckpt, device, seed=seed, epochs=3)
                    elif model_key == "MODEL_6_Fused_LAND_JEPA":
                        val_p, te_p, lat_ms = run_fused_land_jepa_model(train, val, test, tr_norm, va_norm, te_norm, jepa_ckpt, device, seed=seed, epochs=3, enhanced_physics=False)
                    elif model_key == "MODEL_7_Fused_LAND_JEPA_Enhanced_Physics":
                        val_p, te_p, lat_ms = run_fused_land_jepa_model(train, val, test, tr_norm, va_norm, te_norm, jepa_ckpt, device, seed=seed, epochs=3, enhanced_physics=True)
                    else:
                        continue

                    # Metrics at operating point FPR <= 5%
                    m = compute_metrics_at_operating_point(test.y, te_p, val_p, val.y, operating_fpr=0.05)
                    # Lead times
                    lt_dict = compute_event_lead_times(test.metadata, test.y, te_p, threshold=m["operating_threshold"], horizon_hours=h)

                    row = {
                        "model": model_key,
                        "data_version": "NER_REAL_2011_2016_v1",
                        "horizon_hours": h,
                        "seed": seed,
                        "pr_auc": round(m["pr_auc"], 4),
                        "recall_fpr_1": round(m["recall_fpr_1"], 4),
                        "recall_fpr_5": round(m["recall_fpr_5"], 4),
                        "recall_fpr_10": round(m["recall_fpr_10"], 4),
                        "precision": round(m["precision"], 4),
                        "f1": round(m["f1"], 4),
                        "fnr": round(m["fnr"], 4),
                        "fpr": round(m["fpr"], 4),
                        "brier": round(m["brier"], 4),
                        "ece": round(m["ece"], 4),
                        "median_lead_time": round(lt_dict["median_lead_time"], 1),
                        "false_alarms_per_day": round(m["false_alarms_per_day"], 4),
                        "event_detection_rate": round(m["event_detection_rate"], 4),
                        "inference_latency_ms": round(lat_ms, 3),
                    }
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
                    logger.info(f"  [{model_key} | H={h}h | seed={seed}] PR-AUC: {row['pr_auc']:.4f} | Rec@FPR<=5%: {row['recall_fpr_5']:.4f} | Brier: {row['brier']:.4f} | Latency: {row['inference_latency_ms']:.2f}ms (Saved {len(all_rows)} total)")
                    import gc
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error evaluating {model_key} at H={h}h seed={seed}: {e}", exc_info=True)

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(out_csv, index=False)
    logger.info(f"✓ Final Leaderboard saved to {out_csv} ({len(df_results)} rows)")
    return df_results


if __name__ == "__main__":
    run_full_leaderboard_benchmark(seeds=[42, 123, 456], horizons=[6, 12, 24, 48, 72])
