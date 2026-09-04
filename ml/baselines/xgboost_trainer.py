"""
LAND-JEPA — XGBoost Baseline Trainer

Trains an XGBoost binary classifier for landslide risk prediction.
This is the XGBoost BASELINE — a strong traditional ML model used to
contextualise JEPA-TCN improvements.

Design principles:
  1. All hyperparameters from xgboost_config.yaml — zero hard-coded.
  2. scale_pos_weight computed automatically from training class ratio.
  3. Threshold selected on validation set only.
  4. SHAP explanations computed for all predictions.
  5. Model checkpoint includes: model, scaler, feature_names, threshold,
     config_snapshot, git_commit, training_metrics.
  6. No test set touched during training or threshold selection.

Output artifacts (in checkpoint_dir):
  model.ubj           — XGBoost model (JSON format)
  scaler.pkl          — Fitted RobustScaler
  calibrator.pkl      — Fitted isotonic calibrator (optional)
  metadata.json       — Config, metrics, feature names, threshold
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class XGBoostTrainer:
    """
    Trains and evaluates the XGBoost landslide risk baseline.

    Usage:
        trainer = XGBoostTrainer.from_yaml("ml/configs/xgboost_config.yaml")
        result = trainer.train(train_ds, val_ds)
        trainer.save("ml/checkpoints/xgboost")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._model = None
        self._scaler = None
        self._calibrator = None
        self._threshold: float = config.get("training", {}).get("threshold", 0.35)
        self._feature_names: list[str] = []
        self._train_metrics: dict = {}
        self._val_metrics: dict = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "XGBoostTrainer":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "XGBoostTrainer":
        """
        Fit XGBoost on training data, select threshold on validation data.

        Steps:
          1. Fit RobustScaler on X_train.
          2. Compute scale_pos_weight from training class ratio.
          3. Train XGBoost with early stopping on val loss.
          4. Select threshold on val set (F1 strategy).
          5. Compute full metrics on train and val.
          6. Optionally fit isotonic calibrator on val.

        Returns:
            self (for chaining)
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Run: pip install xgboost"
            )

        from ml.preprocessing.normalizers import FeatureNormalizer
        from ml.evaluation.metrics import compute_metrics, select_threshold_on_val
        from ml.evaluation.calibration import (
            fit_isotonic_calibration, calibration_report
        )

        if feature_names:
            self._feature_names = feature_names

        # ── Scale ────────────────────────────────────────────────────
        self._scaler = FeatureNormalizer(scaler_type="robust")
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_val_scaled   = self._scaler.transform(X_val)

        # ── Handle NaN (XGBoost handles NaN natively; still log) ─────
        nan_count_train = int(np.isnan(X_train_scaled).sum())
        if nan_count_train > 0:
            logger.info(
                f"XGBoostTrainer: {nan_count_train} NaN values in training data "
                "(XGBoost handles NaN natively via missing value branching)."
            )

        # ── Class weight ──────────────────────────────────────────────
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos == 0:
            raise ValueError(
                "XGBoostTrainer: No positive samples in training data. "
                "Cannot train a landslide risk classifier."
            )
        scale_pos_weight = n_neg / n_pos
        logger.info(
            f"XGBoostTrainer: n_train={len(y_train)}, "
            f"pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.2f}"
        )

        # ── Build XGBoost params from config ──────────────────────────
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        params = {
            "n_estimators": model_cfg.get("n_estimators", 500),
            "max_depth": model_cfg.get("max_depth", 6),
            "learning_rate": model_cfg.get("learning_rate", 0.05),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "min_child_weight": model_cfg.get("min_child_weight", 5),
            "gamma": model_cfg.get("gamma", 0.1),
            "reg_alpha": model_cfg.get("reg_alpha", 0.1),
            "reg_lambda": model_cfg.get("reg_lambda", 1.0),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": model_cfg.get("tree_method", "hist"),
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "logloss"],
            "seed": model_cfg.get("seed", 42),
            "verbosity": model_cfg.get("verbosity", 0),
            "early_stopping_rounds": train_cfg.get("early_stopping_rounds", 50),
        }

        # ── Train ─────────────────────────────────────────────────────
        self._model = xgb.XGBClassifier(**params)

        # Feature names must be valid identifiers for XGBoost
        safe_names = [f.replace("-", "_").replace(" ", "_") for f in (self._feature_names or [])]

        self._model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
            feature_weights=None,
        )

        n_trees = self._model.best_iteration + 1 if hasattr(self._model, "best_iteration") else params["n_estimators"]
        logger.info(f"XGBoostTrainer: training complete, best_iteration={n_trees}")

        # ── Select threshold on VAL (never on TEST) ───────────────────
        y_prob_val = self._model.predict_proba(X_val_scaled)[:, 1]
        threshold_strategy = train_cfg.get("threshold_strategy", "f1")

        self._threshold = select_threshold_on_val(
            y_true_val=y_val,
            y_prob_val=y_prob_val,
            strategy=threshold_strategy,
        )

        # ── Calibration ───────────────────────────────────────────────
        self._calibrator = fit_isotonic_calibration(y_val, y_prob_val)
        y_prob_val_cal = np.clip(self._calibrator.predict(y_prob_val), 0, 1)
        cal_report = calibration_report(y_val, y_prob_val, y_prob_val_cal)
        logger.info(f"Calibration: {cal_report}")

        # ── Compute metrics (train + val) ─────────────────────────────
        y_prob_train = self._model.predict_proba(X_train_scaled)[:, 1]

        train_result = compute_metrics(
            y_train, y_prob_train, threshold=self._threshold,
            split="train", model_name="xgboost"
        )
        val_result = compute_metrics(
            y_val, y_prob_val, threshold=self._threshold,
            split="val", model_name="xgboost"
        )

        self._train_metrics = train_result.as_dict()
        self._val_metrics = val_result.as_dict()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated risk probabilities for new samples."""
        self._require_trained()
        X_scaled = self._scaler.transform(X)
        raw_proba = self._model.predict_proba(X_scaled)[:, 1]
        if self._calibrator is not None:
            return np.clip(self._calibrator.predict(raw_proba), 0.0, 1.0)
        return raw_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions using the selected threshold."""
        return (self.predict_proba(X) >= self._threshold).astype(int)

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for feature importance.

        Returns:
            SHAP values array of shape (n_samples, n_features).

        DISCLAIMER: SHAP values represent model feature contributions,
        not causal relationships.
        """
        self._require_trained()
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is not installed. Run: pip install shap")

        X_scaled = self._scaler.transform(X)
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X_scaled)
        return shap_values

    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importance scores (gain, weight, or cover)."""
        self._require_trained()
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        if self._feature_names:
            # Map back to original feature names
            safe_to_orig = {
                f.replace("-", "_").replace(" ", "_"): f
                for f in self._feature_names
            }
            return {safe_to_orig.get(k, k): v for k, v in scores.items()}
        return scores

    def evaluate_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Evaluate on test set using threshold selected on val.
        Call ONLY once, after all model selection is complete.

        Returns:
            EvaluationResult as dict.
        """
        from ml.evaluation.metrics import compute_metrics
        self._require_trained()
        y_prob_test = self.predict_proba(X_test)
        result = compute_metrics(
            y_test, y_prob_test,
            threshold=self._threshold,
            split="test",
            model_name="xgboost",
        )
        logger.info("=" * 60)
        logger.info("XGBoost TEST SET EVALUATION (final, reported once)")
        logger.info("=" * 60)
        result.log_summary()
        return result.as_dict()

    def save(self, checkpoint_dir: str | Path) -> Path:
        """
        Save all artifacts to checkpoint_dir.

        Artifacts:
          model.ubj       — XGBoost model (JSON/UBJ binary)
          scaler.pkl      — Fitted RobustScaler
          calibrator.pkl  — Fitted calibrator
          metadata.json   — Config, metrics, features, threshold, git commit
        """
        self._require_trained()
        out = Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Model
        self._model.save_model(str(out / "model.ubj"))

        # Scaler
        self._scaler.save(out / "scaler.pkl")

        # Calibrator
        from ml.evaluation.calibration import save_calibrator
        if self._calibrator is not None:
            save_calibrator(self._calibrator, out / "calibrator.pkl")

        # Metadata
        metadata = {
            "model_name": "xgboost",
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "threshold": self._threshold,
            "feature_names": self._feature_names,
            "config": self.config,
            "train_metrics": self._train_metrics,
            "val_metrics": self._val_metrics,
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"XGBoostTrainer: checkpoint saved to {out}")
        return out

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "XGBoostTrainer":
        """Load a saved XGBoost checkpoint."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is not installed.")

        from ml.preprocessing.normalizers import FeatureNormalizer
        from ml.evaluation.calibration import load_calibrator

        out = Path(checkpoint_dir)

        with open(out / "metadata.json") as f:
            meta = json.load(f)

        trainer = cls(meta["config"])
        trainer._threshold = meta["threshold"]
        trainer._feature_names = meta.get("feature_names", [])
        trainer._train_metrics = meta.get("train_metrics", {})
        trainer._val_metrics = meta.get("val_metrics", {})

        trainer._model = xgb.XGBClassifier()
        trainer._model.load_model(str(out / "model.ubj"))

        trainer._scaler = FeatureNormalizer.load(out / "scaler.pkl")

        calib_path = out / "calibrator.pkl"
        if calib_path.exists():
            trainer._calibrator = load_calibrator(calib_path)

        logger.info(f"XGBoostTrainer: loaded from {out}")
        return trainer

    def _require_trained(self) -> None:
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                "XGBoostTrainer: model not trained yet. Call train() first."
            )
