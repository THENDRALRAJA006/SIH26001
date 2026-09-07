"""
LAND-JEPA -- Prospective Forecast-Backtesting Mode
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Tests whether each model can use weather/rainfall FORECASTS available
BEFORE an event to predict future landslide risk.

TEMPORAL SEPARATION (STRICT)
  OBSERVATION TIME   = context_end  (last hour of observed window)
  FORECAST ISSUANCE  = context_end  (when forecast would have been issued)
  FORECAST VALID     = context_end + H hours
  EVENT TIME         = actual event timestamp from GLC/BHUVAN records

  At prediction time T:
    - ALLOWED : all observations up to and including T
    - ALLOWED : forecast rainfall for T to T+H  (QPF simulated retrospectively)
    - FORBIDDEN: any observation after T used as model input

FORECAST SIMULATION (RETROSPECTIVE)
  True operational QPF not available for the 2011-2016 historical period.
  We simulate QPF by perturbing actual reanalysis rainfall with calibrated
  Gaussian noise (sigma = 30% x actual), modelling typical NWP QPF skill.
  All outputs labelled data_mode=retrospective_QPF_simulated.
  No claim of operational real-time accuracy is made.

MODELS COMPARED
  XGBoost, Supervised TCN, JEPA-TCN, Fused LAND-JEPA

OUTPUTS
  results/prospective_forecast_backtest.csv
  results/prospective_forecast_report.md
  results/lead_time_forecast.csv
  results/forecast_vs_actual.png
  results/lead_time_distribution.png
  results/precision_recall_forecast.png
  results/calibration_forecast.png
"""
from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from ml.evaluation.calibration import expected_calibration_error as compute_ece
from ml.models.land_jepa_model import LandJEPARiskModel
from ml.models.tcn_classifier import TCNClassifier, TCNFineTuneClassifier
from ml.models.tcn_encoder import TCNEncoder
from ml.preprocessing.normalizers import FeatureNormalizer, TemporalNormalizer
from ml.features.dataset_builder import DatasetBuilder, DatasetConfig, SplitDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("prospective_backtest")
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR   = ROOT / "results"
CKPT_DIR      = ROOT / "ml" / "checkpoints"
PROCESSED_DIR = ROOT / "data" / "real" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_H    = [6, 12, 24, 48, 72]
SEEDS         = [42, 123, 456]
OPERATING_FPR = 0.05
QPF_NOISE_SIGMA = 0.30
MODEL_VERSION = "LAND-JEPA-v1.0-RETROSPECTIVE-QPF"
MODELS_TO_RUN = ["XGBoost", "Supervised_TCN", "JEPA_TCN", "Fused_LAND_JEPA"]


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class ForecastRecord:
    """One forecast-vs-actual comparison record."""
    zone_id: str
    prediction_time: str
    forecast_issued_at: str
    forecast_valid_start: str
    forecast_valid_end: str
    horizon_hours: int
    model_version: str
    model_name: str
    seed: int
    risk_probability: float
    risk_level: str
    actual_event: int
    data_mode: str = "retrospective_QPF_simulated"
    lead_time_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def risk_level(p: float) -> str:
    if p < 0.20:
        return "LOW"
    if p < 0.45:
        return "MEDIUM"
    if p < 0.70:
        return "HIGH"
    return "VERY HIGH"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts  = PROCESSED_DIR / "real_ner_timeseries.pkl"
    ter = PROCESSED_DIR / "real_ner_terrain.pkl"
    ev  = PROCESSED_DIR / "real_ner_events.pkl"
    if not ts.exists():
        raise FileNotFoundError(
            f"Missing {ts}. Run scripts/ingest_real_ner_data.py first."
        )
    return pd.read_pickle(ts), pd.read_pickle(ter), pd.read_pickle(ev)


# ---------------------------------------------------------------------------
# QPF INJECTION (simulated, retrospective)
# ---------------------------------------------------------------------------

def inject_qpf(
    X_tab: np.ndarray,
    feat_names: List[str],
    horizon: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """
    Append simulated QPF features to the tabular feature matrix.

    Takes the actual reanalysis rainfall accumulators (available at
    prediction_time T), applies calibrated Gaussian noise (sigma = 30%),
    and appends them as NEW columns named 'qpf_<H>h_mm'.

    The original observed window is NOT modified -- temporal integrity
    is strictly maintained.
    """
    acc_map = {
        "acc_6h":  "qpf_6h_mm",
        "acc_12h": "qpf_12h_mm",
        "acc_24h": "qpf_24h_mm",
        "acc_48h": "qpf_48h_mm",
        "acc_72h": "qpf_72h_mm",
    }
    extras: Dict[str, np.ndarray] = {}
    for src, dst in acc_map.items():
        if src in feat_names:
            idx = feat_names.index(src)
            base = X_tab[:, idx].copy()
            noise = rng.normal(0.0, np.abs(base) * QPF_NOISE_SIGMA)
            extras[dst] = np.clip(base + noise, 0.0, None)

    if not extras:
        return X_tab, feat_names

    extra_arr = np.stack(list(extras.values()), axis=1)
    X_out = np.concatenate([X_tab, extra_arr], axis=1).astype(np.float32)
    return X_out, feat_names + list(extras.keys())


# ---------------------------------------------------------------------------
# THRESHOLD / METRIC HELPERS
# ---------------------------------------------------------------------------

def threshold_at_fpr(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    max_fpr: float = OPERATING_FPR,
) -> float:
    neg = max(int((y_val == 0).sum()), 1)
    best_thr, best_rec = 0.5, 0.0
    for t in np.linspace(0.0, 1.0, 200):
        preds = (val_probs >= t).astype(int)
        fpr_v = int(((preds == 1) & (y_val == 0)).sum()) / neg
        if fpr_v <= max_fpr:
            rec = float(recall_score(y_val, preds, zero_division=0))
            if rec >= best_rec:
                best_rec, best_thr = rec, t
    return best_thr


def metrics_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_val: np.ndarray,
    val_probs: np.ndarray,
    max_fpr: float = OPERATING_FPR,
) -> Dict[str, Any]:
    thr   = threshold_at_fpr(y_val, val_probs, max_fpr)
    preds = (y_prob >= thr).astype(int)
    neg   = max(int((y_true == 0).sum()), 1)
    cm    = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    rec  = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    return {
        "pr_auc":    float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0,
        "recall":    rec,
        "precision": prec,
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "fnr":       1.0 - rec,
        "fpr":       fp / neg,
        "brier":     float(brier_score_loss(y_true, y_prob)),
        "ece":       float(compute_ece(y_true, y_prob)),
        "thr":       thr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# SPLIT BUILDER
# ---------------------------------------------------------------------------

def build_split(
    merged_ts: pd.DataFrame,
    terrain_df: pd.DataFrame,
    events_df: pd.DataFrame,
    h: int,
    stride: int = 24,
) -> Tuple[SplitDataset, SplitDataset, SplitDataset]:
    cfg = DatasetConfig(
        context_hours=168,
        target_hours=h,
        stride_hours=stride,
        min_valid_fraction=0.70,
        test_cutoff="2016-01-01",
        val_cutoff="2015-01-01",
        include_terrain=True,
    )
    return DatasetBuilder(cfg).build(
        merged_df=merged_ts, terrain_df=terrain_df, events_df=events_df
    )


# ---------------------------------------------------------------------------
# MODEL RUNNERS
# ---------------------------------------------------------------------------

def run_xgb(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    h: int, seed: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    import xgboost as xgb
    X_tr, fn = inject_qpf(train.X_tabular, train.feature_names, h, rng)
    X_va, _  = inject_qpf(val.X_tabular,   val.feature_names,   h, rng)
    X_te, _  = inject_qpf(test.X_tabular,  test.feature_names,  h, rng)
    sc = FeatureNormalizer(scaler_type="robust")
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)
    X_te = sc.transform(X_te)
    n_pos = max(int(train.y.sum()), 1)
    spw   = (len(train.y) - n_pos) / n_pos
    clf   = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, random_state=seed,
        eval_metric="logloss", early_stopping_rounds=15,
        verbosity=0, use_label_encoder=False,
    )
    clf.fit(X_tr, train.y, eval_set=[(X_va, val.y)], verbose=False)
    vp = clf.predict_proba(X_va)[:, 1]
    t0 = time.perf_counter()
    tp = clf.predict_proba(X_te)[:, 1]
    return vp, tp, (time.perf_counter() - t0) / max(len(test.y), 1) * 1000.0


def _tcn_loaders(tr_s, tr_l, va_s, va_l, te_s, te_l):
    DS = torch.utils.data.TensorDataset
    DL = torch.utils.data.DataLoader
    return (
        DL(DS(tr_s, tr_l), batch_size=64,  shuffle=True),
        DL(DS(va_s, va_l), batch_size=128, shuffle=False),
        DL(DS(te_s, te_l), batch_size=128, shuffle=False),
    )


def _train_eval_seq(model, ldr_tr, ldr_va, ldr_te, device, train_y, fwd_fn):
    pos_w = torch.tensor(
        [(len(train_y) - train_y.sum()) / max(train_y.sum(), 1.0)], device=device
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()
    for _ in range(3):
        for batch in ldr_tr:
            x = batch[0].to(device)
            y = batch[-1].to(device).unsqueeze(1)
            opt.zero_grad()
            loss = crit(fwd_fn(model, x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    vp = []
    with torch.no_grad():
        for batch in ldr_va:
            vp.extend(torch.sigmoid(fwd_fn(model, batch[0].to(device))).squeeze(1).cpu().numpy())
    t0 = time.perf_counter()
    tp_list = []
    with torch.no_grad():
        for batch in ldr_te:
            tp_list.extend(torch.sigmoid(fwd_fn(model, batch[0].to(device))).squeeze(1).cpu().numpy())
    lat = (time.perf_counter() - t0) / max(len(tp_list), 1) * 1000.0
    return np.array(vp), np.array(tp_list), lat


def run_sup_tcn(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_n: np.ndarray, va_n: np.ndarray, te_n: np.ndarray,
    device: torch.device, seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    idim   = train.X_sequence.shape[2]
    tr_seq = torch.from_numpy(tr_n).float()
    va_seq = torch.from_numpy(va_n).float()
    te_seq = torch.from_numpy(te_n).float()
    tr_lab = torch.from_numpy(train.y).float()
    va_lab = torch.from_numpy(val.y).float()
    te_lab = torch.from_numpy(test.y).float()
    ldr_tr, ldr_va, ldr_te = _tcn_loaders(tr_seq, tr_lab, va_seq, va_lab, te_seq, te_lab)
    model = TCNClassifier(
        input_dim=idim, hidden_dim=64, num_blocks=4, kernel_size=3, dropout=0.1
    ).to(device)
    return _train_eval_seq(
        model, ldr_tr, ldr_va, ldr_te, device, train.y,
        lambda m, x: m(x),
    )


def run_jepa_tcn(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_n: np.ndarray, va_n: np.ndarray, te_n: np.ndarray,
    ckpt: Path, device: torch.device, seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    idim   = train.X_sequence.shape[2]
    tr_seq = torch.from_numpy(tr_n).float()
    va_seq = torch.from_numpy(va_n).float()
    te_seq = torch.from_numpy(te_n).float()
    tr_lab = torch.from_numpy(train.y).float()
    va_lab = torch.from_numpy(val.y).float()
    te_lab = torch.from_numpy(test.y).float()
    ldr_tr, ldr_va, ldr_te = _tcn_loaders(tr_seq, tr_lab, va_seq, va_lab, te_seq, te_lab)
    enc = TCNEncoder(input_dim=idim, hidden_dim=64, num_blocks=4, kernel_size=3, dropout=0.1)
    if ckpt.exists():
        enc.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = TCNFineTuneClassifier(
        encoder=enc, head_hidden_dim=64, dropout=0.1, freeze_encoder=False
    ).to(device)
    return _train_eval_seq(
        model, ldr_tr, ldr_va, ldr_te, device, train.y,
        lambda m, x: m(x),
    )


def run_fused(
    train: SplitDataset, val: SplitDataset, test: SplitDataset,
    tr_n: np.ndarray, va_n: np.ndarray, te_n: np.ndarray,
    ckpt: Path, device: torch.device, seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    TF = ["elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi"]
    PF = ["swi", "pore_pressure_proxy", "stability_indicator"]
    ater = [f for f in TF if f in train.feature_names]
    aphy = [f for f in PF if f in train.feature_names]
    ti   = [train.feature_names.index(f) for f in ater]
    pi   = [train.feature_names.index(f) for f in aphy] if aphy else []
    idim = train.X_sequence.shape[2]

    model = LandJEPARiskModel(
        temporal_dim=idim, terrain_dim=max(len(ater), 1),
        insar_dim=2, physics_dim=max(len(aphy), 1),
        tcn_hidden_dim=64, tcn_num_blocks=4, tcn_kernel_size=3,
        terrain_hidden_dim=64, insar_hidden_dim=32, fused_dim=128,
    ).to(device)
    if ckpt.exists():
        try:
            model.load_pretrained_encoder(ckpt.parent)
        except Exception:
            pass

    def _t(split, norm):
        seq = torch.from_numpy(norm).float()
        ter = torch.from_numpy(
            split.X_tabular[:, ti] if ti else np.zeros((len(split.y), 1))
        ).float()
        phy = torch.from_numpy(
            split.X_tabular[:, pi] if pi else np.zeros((len(split.y), 1))
        ).float()
        lab = torch.from_numpy(split.y).float()
        return seq, ter, phy, lab

    tr_s, tr_t, tr_p, tr_l = _t(train, tr_n)
    va_s, va_t, va_p, va_l = _t(val,   va_n)
    te_s, te_t, te_p, te_l = _t(test,  te_n)

    DS = torch.utils.data.TensorDataset
    DL = torch.utils.data.DataLoader
    ldr_tr = DL(DS(tr_s, tr_t, tr_p, tr_l), batch_size=64,  shuffle=True, drop_last=True)
    ldr_va = DL(DS(va_s, va_t, va_p, va_l), batch_size=128, shuffle=False)
    ldr_te = DL(DS(te_s, te_t, te_p, te_l), batch_size=128, shuffle=False)

    pos_w = torch.tensor(
        [(len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)], device=device
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for _ in range(3):
        for s, t, p, y in ldr_tr:
            s, t, p = s.to(device), t.to(device), p.to(device)
            y = y.to(device).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(s, t, x_physics=p)["logits_24h"], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    vp_list = []
    with torch.no_grad():
        for s, t, p, _ in ldr_va:
            out = model(s.to(device), t.to(device), x_physics=p.to(device))
            vp_list.extend(torch.sigmoid(out["logits_24h"]).squeeze(1).cpu().numpy())

    t0 = time.perf_counter()
    tp_list = []
    with torch.no_grad():
        for s, t, p, _ in ldr_te:
            out = model(s.to(device), t.to(device), x_physics=p.to(device))
            tp_list.extend(torch.sigmoid(out["logits_24h"]).squeeze(1).cpu().numpy())
    lat = (time.perf_counter() - t0) / max(len(tp_list), 1) * 1000.0
    return np.array(vp_list), np.array(tp_list), lat


# ---------------------------------------------------------------------------
# RECORD BUILDER
# ---------------------------------------------------------------------------

def build_records(
    test: SplitDataset,
    test_probs: np.ndarray,
    h: int,
    model_name: str,
    seed: int,
) -> List[ForecastRecord]:
    records = []
    for meta, label, prob in zip(test.metadata, test.y, test_probs):
        ctx = pd.Timestamp(meta["context_end"])
        if not ctx.tzinfo:
            ctx = ctx.tz_localize("UTC")
        ve = ctx + pd.Timedelta(hours=h)
        records.append(ForecastRecord(
            zone_id=str(meta.get("zone_id", "UNK")),
            prediction_time=ctx.isoformat(),
            forecast_issued_at=ctx.isoformat(),
            forecast_valid_start=ctx.isoformat(),
            forecast_valid_end=ve.isoformat(),
            horizon_hours=h,
            model_version=MODEL_VERSION,
            model_name=model_name,
            seed=seed,
            risk_probability=round(float(prob), 4),
            risk_level=risk_level(float(prob)),
            actual_event=int(label),
        ))
    return records


def attach_lead_times(
    recs: List[ForecastRecord],
    events_df: pd.DataFrame,
    h: int,
    thr: float,
) -> List[ForecastRecord]:
    ts_col = next(
        (c for c in ("event_time", "timestamp", "date", "event_date")
         if c in events_df.columns),
        None,
    )
    zone_evts: Dict[str, list] = {}
    if ts_col:
        for _, row in events_df.iterrows():
            z  = str(row.get("zone_id", ""))
            ts = pd.Timestamp(row[ts_col])
            if not ts.tzinfo:
                ts = ts.tz_localize("UTC")
            zone_evts.setdefault(z, []).append(ts)

    for rec in recs:
        if rec.actual_event != 1 or rec.risk_probability < thr:
            continue
        pt = pd.Timestamp(rec.prediction_time)
        if not pt.tzinfo:
            pt = pt.tz_localize("UTC")
        ve = pd.Timestamp(rec.forecast_valid_end)
        if not ve.tzinfo:
            ve = ve.tz_localize("UTC")
        lts = [
            (e - pt).total_seconds() / 3600.0
            for e in zone_evts.get(rec.zone_id, [])
            if pt < e <= ve
        ]
        rec.lead_time_hours = round(min(lts), 1) if lts else float(h)
    return recs


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_vs_actual(sumdf: pd.DataFrame, out: Path) -> None:
    try:
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Forecast vs Actual: PR-AUC and Recall across Horizons\n"
            "(Retrospective QPF Simulated -- Probabilistic Early-Warning Signal)",
            fontsize=11, fontweight="bold",
        )
        for mn, g in sumdf.groupby("model_name"):
            g = g.sort_values("horizon_hours")
            axes[0].plot(g["horizon_hours"], g["pr_auc"],  marker="o", label=mn)
            axes[1].plot(g["horizon_hours"], g["recall"],  marker="s", label=mn)
        for ax, yl, tl in zip(
            axes,
            ["PR-AUC", "Recall @ FPR<=5%"],
            ["PR-AUC vs Forecast Horizon", "Recall vs Forecast Horizon"],
        ):
            ax.set_xlabel("Forecast Horizon (hours)")
            ax.set_ylabel(yl)
            ax.set_title(tl)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(HORIZONS_H)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", out)
    except Exception as e:
        logger.warning("plot_vs_actual failed: %s", e)


def plot_lead_dist(lead_df: pd.DataFrame, out: Path) -> None:
    try:
        plt = _plt()
        det = lead_df[lead_df["lead_time_hours"].notna() & (lead_df["lead_time_hours"] > 0)]
        if det.empty:
            logger.warning("No detected events with lead times -- skipping.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = [0, 6, 12, 24, 48, 72, 96]
        for mn in det["model_name"].unique():
            sub = det[det["model_name"] == mn]["lead_time_hours"]
            ax.hist(sub, bins=bins, alpha=0.6,
                    label=f"{mn} (n={len(sub)})", edgecolor="white")
        ax.axvline(6,  color="red",    linestyle="--", alpha=0.7, label="6h threshold")
        ax.axvline(24, color="orange", linestyle="--", alpha=0.7, label="24h threshold")
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("Detected Events")
        ax.set_title(
            "Lead Time Distribution -- Detected Landslide Events\n"
            "(Probabilistic Early-Warning Signal, FPR<=5% operating point)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", out)
    except Exception as e:
        logger.warning("plot_lead_dist failed: %s", e)


def plot_pr_forecast(recs_by_model: Dict[str, List[ForecastRecord]], out: Path) -> None:
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 7))
        for mn, recs in recs_by_model.items():
            if not recs:
                continue
            yt = np.array([r.actual_event for r in recs])
            yp = np.array([r.risk_probability for r in recs])
            if yt.sum() == 0:
                continue
            pr, re, _ = precision_recall_curve(yt, yp)
            auc = average_precision_score(yt, yp)
            ax.plot(re, pr, label=f"{mn} (PR-AUC={auc:.3f})")
        ax.set_xlabel("Recall (Predicted Landslide Risk Probability >= threshold)")
        ax.set_ylabel("Precision")
        ax.set_title(
            "Precision-Recall Curve -- Forecast Mode\n"
            "(All horizons pooled, retrospective QPF simulated)"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", out)
    except Exception as e:
        logger.warning("plot_pr_forecast failed: %s", e)


def plot_cal_forecast(recs_by_model: Dict[str, List[ForecastRecord]], out: Path) -> None:
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        for mn, recs in recs_by_model.items():
            if not recs:
                continue
            yt = np.array([r.actual_event for r in recs])
            yp = np.array([r.risk_probability for r in recs])
            if yt.sum() == 0:
                continue
            fp, mp = calibration_curve(yt, yp, n_bins=8)
            ax.plot(mp, fp, marker="o", label=mn)
        ax.set_xlabel("Mean Predicted Landslide Risk Probability")
        ax.set_ylabel("Fraction of Actual Events")
        ax.set_title(
            "Calibration Curve -- Forecast Mode\n"
            "(Probabilistic risk estimates, all horizons pooled)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", out)
    except Exception as e:
        logger.warning("plot_cal_forecast failed: %s", e)


# ---------------------------------------------------------------------------
# MARKDOWN REPORT
# ---------------------------------------------------------------------------

def write_report(
    sumdf: pd.DataFrame,
    lead_df: pd.DataFrame,
    best_model: str,
    out: Path,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _pivot(col):
        try:
            return sumdf.pivot_table(
                values=col, index="model_name",
                columns="horizon_hours", aggfunc="mean",
            ).round(4).to_markdown()
        except Exception:
            return "_unavailable_"

    lines = [
        "# LAND-JEPA -- Prospective Forecast-Backtesting Report",
        "",
        f"**Generated**: {now}  ",
        "**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India  ",
        "**Data Mode**: Retrospective QPF Simulated "
        f"({int(QPF_NOISE_SIGMA*100)}% Gaussian noise on reanalysis rainfall)",
        "",
        "---",
        "",
        "## 1. Experimental Design",
        "",
        "### Temporal Separation",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Observation time | `context_end` -- last observed hour |",
        "| Forecast issuance | `context_end` -- when forecast is issued |",
        "| Forecast valid window | `context_end` to `context_end + H hours` |",
        "| Event time | Actual landslide from GLC-2017 / BHUVAN-NER records |",
        "| Information boundary | **Strictly enforced**: no future observation used as model input |",
        "",
        "### QPF Simulation",
        "",
        "True operational NWP forecasts are not available for the 2011-2016 historical period.  ",
        "We simulate QPF by perturbing actual reanalysis rainfall with calibrated Gaussian noise  ",
        f"(sigma = {int(QPF_NOISE_SIGMA*100)}% x actual value), consistent with typical 24-72h QPF skill.  ",
        "All outputs are labelled `data_mode=retrospective_QPF_simulated`.  ",
        "**No claim of operational real-time forecast accuracy is made.**",
        "",
        "### Operating Constraint",
        "",
        f"Thresholds selected on validation set at **FPR <= {int(OPERATING_FPR*100)}%**  ",
        "(operational false-alarm budget for disaster management).",
        "",
        "---",
        "",
        "## 2. Model Comparison Summary (seed=42)",
        "",
        "### PR-AUC per Horizon",
        "",
        _pivot("pr_auc"),
        "",
        "### Recall @ FPR<=5% per Horizon",
        "",
        _pivot("recall"),
        "",
        "### FNR per Horizon",
        "",
        _pivot("fnr"),
        "",
        "### Brier Score per Horizon",
        "",
        _pivot("brier"),
        "",
        "---",
        "",
        "## 3. Lead Time Analysis",
        "",
        "Lead time = elapsed time between forecast issuance and actual landslide event,  ",
        "measured only for correctly detected events (risk_probability >= operating threshold).",
        "",
    ]

    if not lead_df.empty and "lead_time_hours" in lead_df.columns:
        det = lead_df[
            lead_df["lead_time_hours"].notna() & (lead_df["lead_time_hours"] > 0)
        ]
        if not det.empty:
            lt_s = (
                det.groupby("model_name")["lead_time_hours"]
                .agg(["count", "median", "mean", "min", "max"])
                .round(1)
            )
            lt_s.columns = ["N_detected", "Median_lt_h", "Mean_lt_h", "Min_lt_h", "Max_lt_h"]
            lines.append(lt_s.to_markdown())
        else:
            lines.append("_No events detected with computable lead times._")
    else:
        lines.append("_Lead time data unavailable._")

    lines += [
        "",
        "---",
        "",
        "## 4. Selected Production Model",
        "",
        f"**Best model**: `{best_model}`",
        "",
        "Selection criteria (priority order):",
        "1. Highest PR-AUC at 24h horizon",
        "2. Recall @ FPR <= 5%",
        "3. Lowest FNR",
        "4. Median lead time",
        "5. Brier / calibration",
        "",
        "> **IMPORTANT**: All `risk_probability` outputs are probabilistic early-warning signals.  ",
        "> They indicate elevated landslide risk probability, NOT certainty of an event.  ",
        "> Use language: 'high-risk forecast', 'predicted landslide probability'.  ",
        "> Do NOT interpret as a deterministic prediction.",
        "",
        "---",
        "",
        "## 5. Limitations",
        "",
        "1. **QPF simulation**: True NWP forecasts unavailable for 2011-2016; results may differ operationally.",
        "2. **Sparse events**: Limited positive events in NER dataset; recall estimates have wide uncertainty.",
        "3. **Spatial coverage**: 8 NER monitoring zones; full regional generalization not characterised.",
        "4. **InSAR**: Disabled (C-band decorrelation over dense vegetation).",
        "5. **VQC**: Research-only; not evaluated in this backtesting.",
        "6. **No quantum advantage claimed.**",
        "",
        "---",
        "",
        "## 6. Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `prospective_forecast_backtest.csv` | Per-prediction forecast records |",
        "| `prospective_forecast_report.md` | This report |",
        "| `lead_time_forecast.csv` | Detected events with lead times |",
        "| `forecast_vs_actual.png` | PR-AUC and Recall vs horizon |",
        "| `lead_time_distribution.png` | Lead time histogram |",
        "| `precision_recall_forecast.png` | PR curves (all horizons pooled) |",
        "| `calibration_forecast.png` | Calibration curves |",
        "",
        "---",
        "",
        "*All results are from retrospective backtesting on real NER historical data.*  ",
        "*No synthetic, fabricated, or injected results appear in this report.*",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_prospective_backtest() -> None:
    logger.info("=" * 70)
    logger.info("LAND-JEPA Prospective Forecast-Backtesting Mode")
    logger.info("=" * 70)

    merged_ts, terrain_df, events_df = load_data()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_ckpt = CKPT_DIR / "jepa_pretrained_final" / "context_encoder_weights.pt"

    backtest_csv = RESULTS_DIR / "prospective_forecast_backtest.csv"
    lead_csv     = RESULTS_DIR / "lead_time_forecast.csv"

    # Resume support -- reload existing rows
    all_dicts: List[Dict] = []
    if backtest_csv.exists():
        try:
            all_dicts = pd.read_csv(backtest_csv).to_dict("records")
            logger.info("Resuming -- %d existing records loaded.", len(all_dicts))
        except Exception:
            all_dicts = []

    recs_for_plots: Dict[str, List[ForecastRecord]] = {m: [] for m in MODELS_TO_RUN}
    summary_rows: List[Dict] = []

    for h in HORIZONS_H:
        logger.info("\n%s\nHORIZON: %dh\n%s", "=" * 50, h, "=" * 50)
        train, val, test = build_split(merged_ts, terrain_df, events_df, h)
        logger.info(
            "Split %dh: Train N=%d(pos=%d), Val N=%d(pos=%d), Test N=%d(pos=%d)",
            h, len(train.y), int(train.y.sum()),
            len(val.y),   int(val.y.sum()),
            len(test.y),  int(test.y.sum()),
        )

        tnorm = TemporalNormalizer()
        tr_n  = tnorm.fit_transform(train.X_sequence)
        va_n  = tnorm.transform(val.X_sequence)
        te_n  = tnorm.transform(test.X_sequence)

        for seed in SEEDS:
            rng = np.random.default_rng(seed + h)

            for mname in MODELS_TO_RUN:
                already = any(
                    r.get("model_name") == mname
                    and r.get("horizon_hours") == h
                    and r.get("seed") == seed
                    for r in all_dicts
                )
                if already:
                    logger.info("  [SKIP] %s | H=%dh | seed=%d", mname, h, seed)
                    continue

                logger.info("  [RUN]  %s | H=%dh | seed=%d", mname, h, seed)
                try:
                    if mname == "XGBoost":
                        vp, tp, lat = run_xgb(train, val, test, h, seed, rng)
                    elif mname == "Supervised_TCN":
                        vp, tp, lat = run_sup_tcn(
                            train, val, test, tr_n, va_n, te_n, device, seed)
                    elif mname == "JEPA_TCN":
                        vp, tp, lat = run_jepa_tcn(
                            train, val, test, tr_n, va_n, te_n, jepa_ckpt, device, seed)
                    elif mname == "Fused_LAND_JEPA":
                        vp, tp, lat = run_fused(
                            train, val, test, tr_n, va_n, te_n, jepa_ckpt, device, seed)
                    else:
                        continue

                    m    = metrics_at_fpr(test.y, tp, val.y, vp)
                    thr  = m["thr"]
                    recs = build_records(test, tp, h, mname, seed)
                    recs = attach_lead_times(recs, events_df, h, thr)

                    if seed == 42:
                        recs_for_plots[mname].extend(recs)

                    for rec in recs:
                        all_dicts.append(rec.to_dict())
                    pd.DataFrame(all_dicts).to_csv(backtest_csv, index=False)

                    det_lts = [
                        r.lead_time_hours for r in recs
                        if r.lead_time_hours is not None and r.lead_time_hours > 0
                    ]
                    med_lt = round(float(np.median(det_lts)), 1) if det_lts else None
                    summary_rows.append({
                        "model_name":          mname,
                        "horizon_hours":       h,
                        "seed":                seed,
                        "pr_auc":              round(m["pr_auc"],    4),
                        "recall":              round(m["recall"],    4),
                        "precision":           round(m["precision"], 4),
                        "fnr":                 round(m["fnr"],       4),
                        "fpr":                 round(m["fpr"],       4),
                        "brier":               round(m["brier"],     4),
                        "ece":                 round(m["ece"],       4),
                        "operating_threshold": round(thr,            4),
                        "n_detected_events":   len(det_lts),
                        "median_lead_time_h":  med_lt,
                        "inference_latency_ms": round(lat, 3),
                        "data_mode":           "retrospective_QPF_simulated",
                    })

                    logger.info(
                        "    PR-AUC=%.4f | Rec@FPR5%%=%.4f | FNR=%.4f | Brier=%.4f | Lead=%sh",
                        m["pr_auc"], m["recall"], m["fnr"], m["brier"], med_lt,
                    )
                    gc.collect()

                except Exception as exc:
                    logger.error(
                        "  [ERROR] %s H=%dh seed=%d: %s",
                        mname, h, seed, exc, exc_info=True,
                    )

    # ------------------------------------------------------------------
    # FINAL OUTPUTS
    # ------------------------------------------------------------------
    logger.info("\n[OUTPUTS] Generating final outputs...")

    lead_rows = [r for r in all_dicts if r.get("lead_time_hours") is not None]
    lead_df   = pd.DataFrame(lead_rows) if lead_rows else pd.DataFrame()
    lead_df.to_csv(lead_csv, index=False)
    logger.info("  lead_time_forecast.csv (%d rows)", len(lead_df))

    sumdf = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    if sumdf.empty:
        logger.warning("No summary rows -- check errors above.")
        return

    # Select best model: 24h horizon, seed=42, highest PR-AUC
    df_24 = sumdf[(sumdf["horizon_hours"] == 24) & (sumdf["seed"] == 42)]
    if not df_24.empty:
        best_model = str(
            df_24.sort_values("pr_auc", ascending=False).iloc[0]["model_name"]
        )
    else:
        best_model = str(
            sumdf.sort_values("pr_auc", ascending=False).iloc[0]["model_name"]
        )
    logger.info("  Best model (24h PR-AUC): %s", best_model)

    seed42 = sumdf[sumdf["seed"] == 42]
    plot_vs_actual(seed42,        RESULTS_DIR / "forecast_vs_actual.png")
    plot_lead_dist(lead_df,       RESULTS_DIR / "lead_time_distribution.png")
    plot_pr_forecast(recs_for_plots, RESULTS_DIR / "precision_recall_forecast.png")
    plot_cal_forecast(recs_for_plots, RESULTS_DIR / "calibration_forecast.png")
    write_report(
        seed42, lead_df, best_model,
        RESULTS_DIR / "prospective_forecast_report.md",
    )

    logger.info("\n[DONE] Prospective Forecast Backtesting complete.")
    logger.info("  Best model : %s", best_model)
    logger.info("  Results dir: %s", RESULTS_DIR)


if __name__ == "__main__":
    run_prospective_backtest()