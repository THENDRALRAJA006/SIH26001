"""
LAND-JEPA — JEPA Pre-training Script

Runs the full self-supervised JEPA pre-training pipeline:
  1. Generate demo time-series features (no labels needed)
  2. Build context + target window pairs (WindowGenerator)
  3. Normalize with TemporalNormalizer (fit on train)
  4. Run JEPA pre-training with EMA target encoder
  5. Monitor for representation collapse
  6. Save checkpoint

Usage:
    python scripts/pretrain_jepa.py [--config ml/configs/jepa_config.yaml]
                                    [--seed 42]
                                    [--fast]   # 2 epochs, tiny model, for CI

After pre-training, run downstream evaluation with:
    python scripts/evaluate_downstream.py --jepa-checkpoint ml/checkpoints/jepa_pretrained
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def build_jepa_windows(zone_ids, start, end, seed, stride_hours=24):
    """Generate context + target window arrays for JEPA pre-training."""
    import numpy as np
    import pandas as pd
    from ml.ingestion.demo.rainfall_demo import DemoRainfallProvider
    from ml.ingestion.demo.weather_demo import DemoWeatherProvider
    from ml.features.rainfall_features import compute_rainfall_features
    from ml.features.physics_state import PhysicsStateEstimator
    from ml.features.window_generator import WindowConfig, WindowGenerator

    logger.info("Generating demo rainfall + weather for JEPA pre-training...")

    rain = DemoRainfallProvider({"seed": seed})
    rain_df = asyncio.run(rain.fetch(zone_ids, start, end))
    rain_df = rain.validate(rain_df)
    rain_df, _ = rain.transform(rain_df)
    rain_df = compute_rainfall_features(rain_df)

    wx = DemoWeatherProvider({"seed": seed})
    wx_df = asyncio.run(wx.fetch(zone_ids, start, end))
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

    # Generate window pairs
    win_cfg = WindowConfig(
        context_hours=168,
        target_hours=24,
        stride_hours=stride_hours,
        min_valid_fraction=0.70,
    )
    gen = WindowGenerator(win_cfg)
    contexts, targets, meta = gen.generate_arrays(merged)

    logger.info(
        f"JEPA windows: N={len(meta)}, "
        f"ctx_shape={contexts.shape}, tgt_shape={targets.shape}"
    )
    return contexts, targets, meta


def main(args: argparse.Namespace) -> None:
    import numpy as np
    import yaml
    from ml.preprocessing.normalizers import TemporalNormalizer
    from ml.training.jepa_dataset import JEPAPretrainDataset, make_jepa_dataloaders
    from ml.training.jepa_trainer import JEPATrainer

    logger.info("=" * 70)
    logger.info("LAND-JEPA — JEPA Self-Supervised Pre-Training")
    logger.info(f"seed={args.seed}")
    logger.warning(
        "Using DEMO DATA — results are for software integration testing. "
        "Pre-trained representations are NOT validated for operational use."
    )
    logger.info("=" * 70)

    end   = datetime(2023, 9, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=3 * 365)

    stride = 168 if args.fast else 24   # fast: 1 window/week vs 1/day

    contexts, targets, meta = build_jepa_windows(
        DEMO_ZONE_IDS, start, end, args.seed, stride_hours=stride
    )

    if len(meta) == 0:
        logger.error("No windows generated. Check date range and window config.")
        sys.exit(1)

    # ── Temporal split (same boundaries as supervised) ─────────────────
    import pandas as pd
    context_ends = pd.to_datetime(
        [m["context_end"] for m in meta], utc=True
    )
    test_cut  = pd.Timestamp("2023-07-01", tz="UTC")
    val_cut   = pd.Timestamp("2023-01-01", tz="UTC")
    test_mask  = context_ends >= test_cut
    val_mask   = (context_ends >= val_cut) & ~test_mask
    train_mask = ~test_mask & ~val_mask

    idx_train = np.where(train_mask)[0]
    idx_val   = np.where(val_mask)[0]

    if len(idx_train) == 0:
        logger.error("No training windows in date range.")
        sys.exit(1)

    # ── Normalize (fit on train only) ──────────────────────────────────
    norm_ctx = TemporalNormalizer(scaler_type="robust")
    norm_tgt = TemporalNormalizer(scaler_type="robust")

    ctx_train = norm_ctx.fit(contexts[idx_train]).transform(contexts[idx_train])
    tgt_train = norm_tgt.fit(targets[idx_train]).transform(targets[idx_train])

    ctx_val = norm_ctx.transform(contexts[idx_val]) if len(idx_val) > 0 else ctx_train[:10]
    tgt_val = norm_tgt.transform(targets[idx_val]) if len(idx_val) > 0 else tgt_train[:10]

    # Save normalizers
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    norm_ctx.save(ckpt_dir / "context_normalizer.pkl")
    norm_tgt.save(ckpt_dir / "target_normalizer.pkl")

    # ── DataLoaders ────────────────────────────────────────────────────
    with open(args.config) as f:
        import yaml
        cfg = yaml.safe_load(f)

    if args.fast:
        cfg["encoder"]["num_blocks"] = 2
        cfg["training"]["epochs"] = 3
        cfg["training"]["warmup_epochs"] = 1
        cfg["collapse_detection"]["check_interval_epochs"] = 1
    elif args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    batch_size = cfg.get("training", {}).get("batch_size", 128)
    train_ds = JEPAPretrainDataset(ctx_train, tgt_train, augment=True, noise_std=0.01)
    val_ds   = JEPAPretrainDataset(ctx_val,   tgt_val,   augment=False)

    train_loader, val_loader = make_jepa_dataloaders(
        train_ds, val_ds, batch_size=min(batch_size, len(train_ds))
    )

    input_dim = train_ds.n_features
    logger.info(f"input_dim={input_dim}, ctx_len={train_ds.context_len}, tgt_len={train_ds.target_len}")

    # ── Pre-train ──────────────────────────────────────────────────────
    trainer = JEPATrainer(cfg)
    trainer.pretrain(train_loader, val_loader, input_dim=input_dim)
    trainer.save(ckpt_dir)

    logger.info(f"JEPA pre-training complete. Checkpoint: {ckpt_dir}")
    logger.info(
        "Next: run downstream evaluation with:\n"
        f"  python scripts/evaluate_downstream.py "
        f"--jepa-checkpoint {ckpt_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAND-JEPA self-supervised pre-training")
    parser.add_argument("--config", default="ml/configs/jepa_config.yaml")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints/jepa_pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: tiny model, 3 epochs, weekly stride (for CI)")
    args = parser.parse_args()
    main(args)
