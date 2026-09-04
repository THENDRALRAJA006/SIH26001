"""
LAND-JEPA — TCN Baseline Training Script

Runs the supervised TCN training pipeline:
  1. Generate demo features (or load from CSV if pre-generated)
  2. Build labelled dataset with temporal split
  3. Wrap in PyTorch DataLoaders
  4. Normalize with TemporalNormalizer (fit on train only)
  5. Train TCN with early stopping
  6. Compare against XGBoost baseline (if checkpoint exists)
  7. Save checkpoint + experiment log

Usage:
    python scripts/train_tcn.py [--config ml/configs/tcn_config.yaml]
                                [--label-fraction 1.0]
                                [--seed 42]
                                [--evaluate-test]
                                [--fast]           # 2-block tiny model for quick CI
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def main(args: argparse.Namespace) -> None:
    import yaml
    from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
    from ml.features.rainfall_features import compute_rainfall_features
    from ml.features.physics_state import PhysicsStateEstimator
    from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
    from ml.ingestion.demo.weather_demo import DemoWeatherProvider
    from ml.ingestion.demo.terrain_landslide_demo import (
        DemoTerrainProvider, DemoLandslideInventoryProvider
    )
    from ml.preprocessing.normalizers import TemporalNormalizer
    from ml.training.tcn_dataset import LandslideSequenceDataset, make_dataloaders
    from ml.training.tcn_trainer import TCNTrainer
    from ml.evaluation.experiment_logger import ExperimentLogger

    logger.info("=" * 70)
    logger.info("LAND-JEPA — Supervised TCN Baseline Training")
    logger.info(f"label_fraction={args.label_fraction:.0%} | seed={args.seed}")
    logger.warning("Using DEMO DATA — results are for integration testing only.")
    logger.info("=" * 70)

    end   = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=3 * 365)

    # ── Generate features ─────────────────────────────────────────────
    logger.info("Generating demo rainfall features...")
    rain = DemoRainfallProvider({"seed": args.seed})
    rain_df = asyncio.run(rain.fetch(DEMO_ZONE_IDS, start, end))
    rain_df = rain.validate(rain_df)
    rain_df, _ = rain.transform(rain_df)
    rain_df = compute_rainfall_features(rain_df)

    wx = DemoWeatherProvider({"seed": args.seed})
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

    terrain_p = DemoTerrainProvider({"seed": args.seed})
    terrain_df = asyncio.run(terrain_p.fetch(DEMO_ZONE_IDS, start, end))
    terrain_df = terrain_p.validate(terrain_df)
    terrain_df, _ = terrain_p.transform(terrain_df)

    events_p = DemoLandslideInventoryProvider({"seed": args.seed, "n_events": 100})
    events_df = asyncio.run(events_p.fetch(DEMO_ZONE_IDS, start, end))
    events_df = events_p.validate(events_df)
    events_df, _ = events_p.transform(events_df)

    # ── Build dataset ─────────────────────────────────────────────────
    ds_cfg = DatasetConfig(
        context_hours=168,
        target_hours=24,
        stride_hours=48 if not args.fast else 168,
        min_valid_fraction=0.70,
        test_cutoff="2023-07-01",
        val_cutoff="2023-01-01",
    )
    builder = DatasetBuilder(ds_cfg)
    train_ds, val_ds, test_ds = builder.build(
        merged_df=merged,
        terrain_df=terrain_df,
        events_df=events_df,
        label_fraction=args.label_fraction,
        label_seed=args.seed,
    )

    if train_ds.n_samples == 0:
        logger.error("No training windows. Check date range and window config.")
        sys.exit(1)

    logger.info(f"Train: {train_ds.describe()}")
    logger.info(f"Val:   {val_ds.describe()}")
    logger.info(f"Test:  {test_ds.describe()}")

    # ── Normalize sequences (fit on train only) ───────────────────────
    normalizer = TemporalNormalizer(scaler_type="robust")
    X_train_norm = normalizer.fit(train_ds.X_sequence).transform(train_ds.X_sequence)
    X_val_norm   = normalizer.transform(val_ds.X_sequence)
    X_test_norm  = normalizer.transform(test_ds.X_sequence) if test_ds.n_samples > 0 else val_ds.X_sequence

    # Save normalizer
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(ckpt_dir / "sequence_normalizer.pkl")

    # ── Build DataLoaders ─────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override for fast mode
    if args.fast:
        cfg["model"]["num_blocks"] = 2
        cfg["training"]["epochs"] = 5
        cfg["training"]["early_stopping_patience"] = 3

    batch_size = cfg.get("training", {}).get("batch_size", 64)

    train_pt = LandslideSequenceDataset(X_train_norm, train_ds.y, augment=True, noise_std=0.01)
    val_pt   = LandslideSequenceDataset(X_val_norm,   val_ds.y,   augment=False)
    test_pt  = LandslideSequenceDataset(X_test_norm,  test_ds.y,  augment=False)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_pt, val_pt, test_pt, batch_size=batch_size
    )

    input_dim = train_pt.n_features
    logger.info(f"input_dim={input_dim}, seq_len={train_pt.seq_len}")

    # ── Train TCN ─────────────────────────────────────────────────────
    trainer = TCNTrainer(cfg)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        feature_names=train_ds.feature_names,
    )

    # ── Save checkpoint ───────────────────────────────────────────────
    trainer.save(ckpt_dir)

    # ── Log experiment ────────────────────────────────────────────────
    exp_logger = ExperimentLogger("results/experiment_log.jsonl")
    exp_logger.log(
        metrics=trainer._val_metrics,
        config_snapshot=cfg,
        model_name="tcn_supervised",
        notes=f"label_fraction={args.label_fraction:.0%} seed={args.seed} DEMO_DATA",
    )

    # ── Optional test evaluation ──────────────────────────────────────
    if args.evaluate_test:
        if test_ds.n_samples == 0:
            logger.warning("No test samples — skipping.")
        else:
            logger.warning(
                "TEST SET EVALUATION: report ONCE. Do NOT use for tuning."
            )
            test_result = trainer.evaluate_test(test_loader)
            exp_logger.log(
                metrics=test_result, config_snapshot=cfg,
                model_name="tcn_supervised",
                notes=f"TEST SET | label_fraction={args.label_fraction:.0%} | DEMO_DATA",
            )

    exp_logger.summarize()
    logger.info("TCN training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LAND-JEPA supervised TCN")
    parser.add_argument("--config", default="ml/configs/tcn_config.yaml")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints/tcn_supervised")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Quick run with tiny model and few epochs (for CI)")
    args = parser.parse_args()
    main(args)
