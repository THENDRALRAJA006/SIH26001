"""
Run Spatial Generalization (Leave-One-Zone-Out) and Component Ablations (A-D)
Saves results directly to:
  - results/spatial_validation.csv
  - results/ablation.csv
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gis.real_zones import REAL_ZONE_IDS, REAL_NER_ZONES
from ml.features.dataset_builder import DatasetBuilder, DatasetConfig
from ml.preprocessing.normalizers import TemporalNormalizer
from ml.training.tcn_dataset import LandslideSequenceDataset, make_dataloaders
from scripts.run_final_benchmark import (
    load_final_real_data,
    run_ablation,
    leave_one_zone_out,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("spatial_ablation")

def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_ckpt = ROOT / "ml" / "checkpoints" / "jepa_pretrained_final" / "context_encoder_weights.pt"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading real NER data for spatial validation & ablations...")
    builder, merged_ts, terrain_df, events_df = load_final_real_data(include_terrain=True)
    builder.warm_cache(merged_ts, terrain_df, events_df)

    ts_feature_cols = [
        c for c in merged_ts.columns
        if c not in ("zone_id", "observed_at", "data_source", "is_demo", "quality_flag")
    ]
    input_dim = len(ts_feature_cols)

    # 1. Run Phase 14: Ablation Studies A-D
    logger.info("Running Component Ablations (Phase 14)...")
    train_ab, val_ab, test_ab = builder.build_cached(label_fraction=1.0, label_seed=42)
    tnorm_ab = TemporalNormalizer()
    tnorm_ab.fit_transform(train_ab.X_sequence)
    df_ablation = run_ablation(
        train_ab, val_ab, test_ab, tnorm_ab, jepa_ckpt, device, seed=42
    )
    df_ablation.to_csv(results_dir / "ablation.csv", index=False)
    df_ablation.to_csv(results_dir / "ablation_results.csv", index=False)
    logger.info(f"Saved results/ablation.csv ({len(df_ablation)} rows)")
    print(df_ablation[["ablation", "aucpr", "recall", "fnr", "f1", "brier_score", "fpr"]])

    # 2. Run Phase 13: Leave-One-Zone-Out Spatial Validation
    logger.info("Running Spatial Generalization LOZO (Phase 13)...")
    df_lozo = leave_one_zone_out(merged_ts, terrain_df, events_df, input_dim, device, jepa_ckpt)
    df_lozo.to_csv(results_dir / "spatial_validation.csv", index=False)
    df_lozo.to_csv(results_dir / "spatial_generalization_lozo.csv", index=False)
    logger.info(f"Saved results/spatial_validation.csv ({len(df_lozo)} rows)")
    print(df_lozo)

if __name__ == "__main__":
    main()
