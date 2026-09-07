"""
LAND-JEPA — Downstream Evaluation of Pretrained JEPA Representations

Evaluates the pretrained JEPA encoder on downstream landslide risk classification:
  1. Linear Probe (frozen encoder, train classification head only)
  2. Full Fine-Tuning (unfreeze encoder with differential learning rate)
  3. Label-Efficiency Experiment across fractions [10%, 50%, 100%]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.evaluation.downstream_eval import DownstreamEvaluator, run_label_efficiency_experiment
from ml.evaluation.experiment_logger import ExperimentLogger
from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
from ml.features.physics_state import PhysicsStateEstimator
from ml.features.rainfall_features import compute_rainfall_features
from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
from ml.ingestion.demo.terrain_landslide_demo import (
    DemoLandslideInventoryProvider,
    DemoTerrainProvider,
)
from ml.ingestion.demo.weather_demo import DemoWeatherProvider
from ml.preprocessing.normalizers import TemporalNormalizer
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


def load_downstream_data(seed: int, fast: bool = False):
    """Build datasets for downstream evaluation."""
    end = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=3 * 365)

    logger.info("Generating demo environmental data for downstream evaluation...")
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
        stride_hours=24 if not fast else 48,
        min_valid_fraction=0.70,
        test_cutoff="2023-07-01",
        val_cutoff="2023-01-01",
    )
    builder = DatasetBuilder(ds_cfg)
    return builder, merged, terrain_df, events_df


def main(args: argparse.Namespace) -> None:
    logger.info("=" * 70)
    logger.info("LAND-JEPA — Pretrained JEPA Downstream Evaluation")
    logger.info(f"Checkpoint: {args.jepa_checkpoint}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 70)

    builder, merged, terrain_df, events_df = load_downstream_data(args.seed, args.fast)

    # Initial full dataset to fit normalizer
    train_full, val_ds, test_ds = builder.build(
        merged_df=merged,
        terrain_df=terrain_df,
        events_df=events_df,
        label_fraction=1.0,
        label_seed=args.seed,
    )

    normalizer = TemporalNormalizer(scaler_type="robust")
    normalizer.fit(train_full.X_sequence)

    X_val_norm = normalizer.transform(val_ds.X_sequence)
    val_dataset = LandslideSequenceDataset(X_val_norm, val_ds.y, augment=False)
    _, val_loader = make_dataloaders(val_dataset, val_dataset, batch_size=32)

    X_test_norm = normalizer.transform(test_ds.X_sequence) if test_ds.n_samples > 0 else X_val_norm
    test_y = test_ds.y if test_ds.n_samples > 0 else val_ds.y
    test_dataset = LandslideSequenceDataset(X_test_norm, test_y, augment=False)
    _, test_loader = make_dataloaders(test_dataset, test_dataset, batch_size=32)

    input_dim = train_full.X_sequence.shape[-1]
    evaluator = DownstreamEvaluator(
        jepa_checkpoint_dir=args.jepa_checkpoint,
        downstream_config_path=args.config,
        input_dim=input_dim,
    )

    # Helper function for label fraction loaders
    def make_loaders_fn(label_fraction: float, seed: int):
        train_sub, _, _ = builder.build(
            merged_df=merged,
            terrain_df=terrain_df,
            events_df=events_df,
            label_fraction=label_fraction,
            label_seed=seed,
        )
        X_sub_norm = normalizer.transform(train_sub.X_sequence)
        train_dataset = LandslideSequenceDataset(X_sub_norm, train_sub.y, augment=True)
        t_loader, v_loader = make_dataloaders(train_dataset, val_dataset, batch_size=32)
        return t_loader, v_loader

    fractions = [0.10, 0.50, 1.0] if not args.fast else [0.50, 1.0]
    protocols = ["linear_probe", "fine_tune"]
    exp_logger = ExperimentLogger("results/experiment_log.jsonl")

    results = run_label_efficiency_experiment(
        evaluator=evaluator,
        make_loaders_fn=make_loaders_fn,
        label_fractions=fractions,
        seeds=[args.seed],
        protocols=protocols,
        experiment_logger=exp_logger,
    )

    print("\n" + "=" * 80)
    print("DOWNSTREAM TRANSFER & LABEL-EFFICIENCY RESULTS")
    print("=" * 80)
    print(f"{'Protocol':<15} {'Label Frac':<12} {'Val AUCPR':<12} {'Val AUROC':<12} {'Val F1':<10} {'Val ECE':<10}")
    print("-" * 80)
    for r in results:
        vm = r.val_metrics
        print(
            f"{r.protocol:<15} "
            f"{r.label_fraction:<12.0%} "
            f"{vm.get('aucpr', 0.0):<12.4f} "
            f"{vm.get('auroc', 0.0):<12.4f} "
            f"{vm.get('f1', 0.0):<10.4f} "
            f"{vm.get('ece', 0.0):<10.4f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pretrained JEPA downstream")
    parser.add_argument("--jepa-checkpoint", default="ml/checkpoints/jepa_pretrained")
    parser.add_argument("--config", default="ml/configs/downstream_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Fast evaluation run")
    args = parser.parse_args()
    main(args)
