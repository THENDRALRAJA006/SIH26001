"""
LAND-JEPA — XGBoost Training Script

Runs the full XGBoost baseline training pipeline:
  1. Load demo data (or real data if configured)
  2. Compute rainfall + weather features
  3. Compute physics state (SWI)
  4. Build labelled dataset with temporal split
  5. Train XGBoost on training set
  6. Select threshold on validation set
  7. Evaluate on validation set (not test!)
  8. Save checkpoint
  9. Log experiment

Usage:
    python scripts/train_xgboost.py [--config ml/configs/xgboost_config.yaml]
                                    [--data-config ml/configs/data_config.yaml]
                                    [--label-fraction 1.0]
                                    [--seed 42]
                                    [--evaluate-test]   # only if final reporting

  --evaluate-test: ONLY use this flag for the FINAL reporting run.
                   Do NOT use it during hyperparameter tuning or model selection.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Demo zone IDs (must match generate_demo_data.py) ─────────────────
DEMO_ZONE_IDS = [
    "DEMO-NER-001", "DEMO-NER-002", "DEMO-NER-003", "DEMO-NER-004",
    "DEMO-NER-005", "DEMO-NER-006", "DEMO-NER-007", "DEMO-NER-008",
]


def build_demo_features(
    zone_ids: list[str],
    start: datetime,
    end: datetime,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and merge all features from demo providers.

    Returns:
        (merged_df, terrain_df, events_df)
    """
    from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
    from ml.ingestion.demo.weather_demo import DemoWeatherProvider
    from ml.ingestion.demo.terrain_landslide_demo import (
        DemoTerrainProvider, DemoLandslideInventoryProvider
    )
    from ml.features.rainfall_features import compute_rainfall_features
    from ml.features.physics_state import PhysicsStateEstimator, PhysicsConfig

    logger.info("Fetching demo rainfall...")
    rain_provider = DemoRainfallProvider({"seed": seed})
    rain_df = asyncio.run(rain_provider.fetch(zone_ids, start, end))
    rain_df = rain_provider.validate(rain_df)
    rain_df, _ = rain_provider.transform(rain_df)

    logger.info("Computing rolling rainfall features...")
    rain_df = compute_rainfall_features(rain_df)

    logger.info("Fetching demo weather...")
    wx_provider = DemoWeatherProvider({"seed": seed})
    wx_df = asyncio.run(wx_provider.fetch(zone_ids, start, end))
    wx_df = wx_provider.validate(wx_df)
    wx_df, _ = wx_provider.transform(wx_df)

    logger.info("Computing physics state (SWI)...")
    phys = PhysicsStateEstimator()
    phys_df = phys.compute(rain_df[["zone_id", "observed_at", "precipitation_mm"]])

    logger.info("Merging time-series features...")
    # Merge on zone_id + observed_at
    merged = rain_df.merge(
        wx_df[["zone_id", "observed_at", "temperature_c", "humidity_pct", "wind_speed_ms"]],
        on=["zone_id", "observed_at"], how="left"
    )
    merged = merged.merge(
        phys_df[["zone_id", "observed_at", "swi", "pore_pressure_proxy", "stability_indicator"]],
        on=["zone_id", "observed_at"], how="left"
    )
    # Drop metadata columns not needed for ML
    drop_cols = ["data_source", "is_demo", "quality_flag", "precipitation_mm"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    logger.info("Fetching demo terrain...")
    terrain_provider = DemoTerrainProvider({"seed": seed})
    terrain_df = asyncio.run(terrain_provider.fetch(zone_ids, start, end))
    terrain_df = terrain_provider.validate(terrain_df)
    terrain_df, _ = terrain_provider.transform(terrain_df)

    logger.info("Fetching demo landslide events...")
    events_provider = DemoLandslideInventoryProvider({"seed": seed, "n_events": 100})
    events_df = asyncio.run(events_provider.fetch(zone_ids, start, end))
    events_df = events_provider.validate(events_df)
    events_df, _ = events_provider.transform(events_df)

    logger.info(
        f"Data ready: merged={len(merged)} rows, "
        f"terrain={len(terrain_df)} zones, "
        f"events={len(events_df)} events"
    )
    return merged, terrain_df, events_df


def main(args: argparse.Namespace) -> None:
    from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
    from ml.baselines.xgboost_trainer import XGBoostTrainer
    from ml.evaluation.experiment_logger import ExperimentLogger
    from ml.evaluation.metrics import compute_baseline_metrics
    import yaml

    logger.info("=" * 70)
    logger.info("LAND-JEPA — XGBoost Baseline Training")
    logger.info(f"label_fraction={args.label_fraction:.0%} | seed={args.seed}")
    logger.warning(
        "This script uses DEMO DATA. Results are for software integration "
        "testing only, NOT scientific performance claims."
    )
    logger.info("=" * 70)

    # ── Date range: 3 years of demo data ─────────────────────────────
    end   = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=3 * 365)

    # ── Generate features ─────────────────────────────────────────────
    merged_df, terrain_df, events_df = build_demo_features(
        zone_ids=DEMO_ZONE_IDS,
        start=start,
        end=end,
        seed=args.seed,
    )

    # ── Build dataset ─────────────────────────────────────────────────
    ds_cfg = DatasetConfig(
        context_hours=168,  # 7 days
        target_hours=24,
        stride_hours=24,    # 1 window per day (manageable for demo)
        min_valid_fraction=0.70,
        test_cutoff="2023-07-01",
        val_cutoff="2023-01-01",
        label_efficiency_fraction=args.label_fraction,
    )
    builder = DatasetBuilder(ds_cfg)
    train_ds, val_ds, test_ds = builder.build(
        merged_df=merged_df,
        terrain_df=terrain_df,
        events_df=events_df,
        label_fraction=args.label_fraction,
        label_seed=args.seed,
    )

    if train_ds.n_samples == 0:
        logger.error("No training samples generated. Check data date ranges and window config.")
        sys.exit(1)

    logger.info(f"Train: {train_ds.describe()}")
    logger.info(f"Val:   {val_ds.describe()}")
    logger.info(f"Test:  {test_ds.describe()}")

    # Baseline sanity check
    if train_ds.n_samples > 0:
        baseline = compute_baseline_metrics(train_ds.y)
        logger.info(f"Random baseline AUCPR: {baseline['random_aucpr']:.4f}")

    # ── Train XGBoost ─────────────────────────────────────────────────
    trainer = XGBoostTrainer.from_yaml(args.config)
    trainer.train(
        X_train=train_ds.X_tabular,
        y_train=train_ds.y,
        X_val=val_ds.X_tabular if val_ds.n_samples > 0 else train_ds.X_tabular,
        y_val=val_ds.y if val_ds.n_samples > 0 else train_ds.y,
        feature_names=train_ds.feature_names,
    )

    # ── Save checkpoint ───────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    trainer.save(checkpoint_dir)

    # ── Log experiment ────────────────────────────────────────────────
    exp_logger = ExperimentLogger("results/experiment_log.jsonl")
    exp_logger.log(
        metrics=trainer._val_metrics,
        config_snapshot=trainer.config,
        model_name="xgboost",
        notes=f"label_fraction={args.label_fraction:.0%} seed={args.seed} DEMO_DATA",
    )

    # ── Evaluate test (only if explicitly requested) ──────────────────
    if args.evaluate_test:
        if test_ds.n_samples == 0:
            logger.warning("No test samples — skipping test evaluation.")
        else:
            logger.warning(
                "TEST SET EVALUATION: This result should be reported ONCE. "
                "Do NOT use it to tune any parameters."
            )
            test_metrics = trainer.evaluate_test(test_ds.X_tabular, test_ds.y)
            exp_logger.log(
                metrics=test_metrics,
                config_snapshot=trainer.config,
                model_name="xgboost",
                notes=f"TEST SET | label_fraction={args.label_fraction:.0%} | DEMO_DATA",
            )
    else:
        logger.info("Test set not evaluated (use --evaluate-test for final reporting).")

    exp_logger.summarize()
    logger.info("XGBoost training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LAND-JEPA XGBoost baseline")
    parser.add_argument("--config", default="ml/configs/xgboost_config.yaml")
    parser.add_argument("--data-config", default="ml/configs/data_config.yaml")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints/xgboost")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on test set (use ONLY for final reporting, not tuning).",
    )
    args = parser.parse_args()
    main(args)
