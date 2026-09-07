"""
LAND-JEPA — Dataset Builder

Assembles the full ML training dataset from:
  1. Rainfall features (rolling accumulators)
  2. Weather features (temperature, humidity, wind)
  3. Soil moisture features
  4. Physics state features (SWI, pore-pressure)
  5. Static terrain features (elevation, slope, TWI, etc.)
  6. Historical susceptibility proxies
  7. Binary labels + label_confidence from LabelBuilder

Outputs:
  - X_tabular: 2D numpy array (N_windows, N_features) for XGBoost
  - X_sequence: 3D numpy array (N_windows, T, F) for TCN / JEPA
  - y: 1D numpy array of labels {0, 1}   (-1 excluded)
  - feature_names: list of column names matching X_tabular columns
  - metadata: list of dicts {zone_id, context_end, ...}

Split policy (from data_config.yaml):
  - test_cutoff:  context_end >= test_cutoff   → test
  - val_cutoff:   context_end >= val_cutoff     → val
  - else                                        → train
  NEVER shuffle across time boundaries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Mirrors data_config.yaml — all fields configurable."""
    context_hours: int = 168
    target_hours: int = 24
    stride_hours: int = 1
    min_valid_fraction: float = 0.80
    test_cutoff: str = "2022-01-01"
    val_cutoff: str = "2021-01-01"
    label_efficiency_fraction: float = 1.0
    label_efficiency_seed: int = 42
    include_terrain: bool = True
    include_sequence: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        split = cfg.get("split", {})
        le = cfg.get("label_efficiency", {})
        return cls(
            context_hours=cfg.get("window", {}).get("context_hours", 168),
            target_hours=cfg.get("window", {}).get("target_hours", 24),
            test_cutoff=split.get("test_cutoff", "2022-01-01"),
            val_cutoff=split.get("val_cutoff", "2021-01-01"),
            label_efficiency_fraction=le.get("fractions", [1.0])[-1],
            include_terrain=cfg.get("features", {}).get("include_terrain", True),
        )


@dataclass
class SplitDataset:
    """One train/val/test split."""
    X_tabular: np.ndarray          # (N, F)
    X_sequence: np.ndarray         # (N, T, n_ts_features)
    y: np.ndarray                  # (N,) binary {0,1}
    feature_names: list[str]
    metadata: list[dict]
    split_name: str                # 'train' | 'val' | 'test'
    label_fraction: float = 1.0

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) > 0 else 0.0

    def describe(self) -> str:
        return (
            f"[{self.split_name}] N={self.n_samples}, "
            f"pos_rate={self.positive_rate:.3f}, "
            f"n_features={self.X_tabular.shape[1]}, "
            f"label_fraction={self.label_fraction:.0%}"
        )


class DatasetBuilder:
    """
    Builds tabular and sequential datasets from merged feature DataFrames.

    Usage:
        builder = DatasetBuilder(config)
        train, val, test = builder.build(
            rainfall_df=...,
            weather_df=...,
            soil_df=...,
            terrain_df=...,
            physics_df=...,
            events_df=...,
        )
    """

    # Tabular features extracted from the LAST timestep of each context window
    # (representing "current state" for XGBoost)
    # NOTE: sm_anomaly is intentionally omitted — it is not present in ERA5-Land
    # real ingestion and would be silently zero-filled, creating a dead feature.
    SNAPSHOT_FEATURES = [
        "acc_1h", "acc_3h", "acc_6h", "acc_12h", "acc_24h", "acc_48h", "acc_72h",
        "intensity_max_1h", "dry_hours_streak", "monsoon_flag",
        "temperature_c", "humidity_pct", "wind_speed_ms",
        "sm_volumetric",
        "swi", "pore_pressure_proxy", "stability_indicator",
    ]
    # Static features (same for all windows of the same zone)
    STATIC_FEATURES = [
        "elevation_m", "slope_deg", "aspect_deg", "curvature", "tpi", "twi",
    ]

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        # ── Window cache (populated by build_with_cache) ──────────────────────
        self._cached_contexts: np.ndarray | None = None
        self._cached_labels_full: object = None  # LabelBuilder result at fraction=1.0
        self._cached_meta: list | None = None
        self._cached_terrain_lookup: dict | None = None
        self._cached_ts_feature_cols: list | None = None
        self._cached_feat_names: list | None = None
        self._cached_X_tab: np.ndarray | None = None
        self._cached_context_ends_ts: pd.DatetimeIndex | None = None

    # ── Single pass: generate + cache windows ─────────────────────────────────
    def warm_cache(self, merged_df: pd.DataFrame, terrain_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
        """Generate all windows ONCE and store in memory. Call this before the benchmark loop."""
        from ml.features.window_generator import WindowConfig, WindowGenerator
        from ml.features.label_builder import LabelBuilder, LabelConfig

        cfg = self.config
        win_cfg = WindowConfig(
            context_hours=cfg.context_hours, target_hours=cfg.target_hours,
            stride_hours=cfg.stride_hours, min_valid_fraction=cfg.min_valid_fraction,
        )
        gen = WindowGenerator(win_cfg)

        self._cached_ts_feature_cols = [
            c for c in merged_df.columns
            if c not in ("zone_id", "observed_at", "data_source", "is_demo", "quality_flag")
        ]

        contexts_arr, _, meta = gen.generate_arrays(merged_df)
        contexts_arr = contexts_arr.astype(np.float32)
        logger.info(f"DatasetBuilder.warm_cache: {len(meta)} windows generated (cached)")

        # Build labels at fraction=1.0 (full label set)
        builder_lb = LabelBuilder(LabelConfig(event_window_hours=cfg.target_hours))
        labels_full = builder_lb.build(meta, events_df)

        # Build terrain lookup
        terrain_lookup: dict[str, dict] = {}
        for _, row in terrain_df.iterrows():
            terrain_lookup[str(row["zone_id"])] = {
                feat: row.get(feat, np.nan) for feat in self.STATIC_FEATURES
            }

        self._cached_contexts = contexts_arr
        self._cached_meta = meta
        self._cached_labels_full = (builder_lb, labels_full)
        self._cached_terrain_lookup = terrain_lookup
        feat_names = self.SNAPSHOT_FEATURES + (
            self.STATIC_FEATURES if cfg.include_terrain else []
        )
        self._cached_feat_names = feat_names

        # Precompute X_tab for all windows
        static_feats = self.STATIC_FEATURES if cfg.include_terrain else []
        X_tab_rows = []
        for i, m in enumerate(meta):
            ctx = contexts_arr[i]
            snapshot = {
                col: float(ctx[-1, j]) if j < ctx.shape[1] else np.nan
                for j, col in enumerate(self._cached_ts_feature_cols)
                if col in self.SNAPSHOT_FEATURES
            }
            for feat in self.SNAPSHOT_FEATURES:
                if feat not in snapshot:
                    snapshot[feat] = np.nan

            if cfg.include_terrain:
                terrain_row = terrain_lookup.get(m["zone_id"], {})
                for feat in static_feats:
                    snapshot[feat] = terrain_row.get(feat, np.nan)

            x_tab = np.array([snapshot.get(f, np.nan) for f in feat_names], dtype=np.float32)
            X_tab_rows.append(x_tab)

        self._cached_X_tab = np.stack(X_tab_rows) if X_tab_rows else np.empty((0, len(feat_names)), dtype=np.float32)
        self._cached_context_ends_ts = pd.to_datetime([m["context_end"] for m in meta], utc=True)

    def build_cached(
        self,
        label_fraction: float = 1.0,
        label_seed: int = 42,
    ) -> tuple["SplitDataset", "SplitDataset", "SplitDataset"]:
        """Fast build using cached windows. Call warm_cache() once first."""
        if self._cached_contexts is None or self._cached_X_tab is None or self._cached_context_ends_ts is None:
            raise RuntimeError("Call warm_cache() before build_cached()")

        cfg = self.config
        builder_lb, labels_base = self._cached_labels_full  # type: ignore
        meta = self._cached_meta  # type: ignore
        feat_names = self._cached_feat_names  # type: ignore
        test_cutoff = pd.Timestamp(cfg.test_cutoff, tz="UTC")
        val_cutoff  = pd.Timestamp(cfg.val_cutoff,  tz="UTC")

        # Apply fraction subsampling (fast — just masks some positives)
        labels_df = labels_base.copy()
        if label_fraction < 1.0:
            labels_df = builder_lb.apply_label_efficiency_fraction(
                labels_df, fraction=label_fraction, seed=label_seed
            )

        labels_arr = labels_df["label"].to_numpy(dtype=np.int32)
        valid_mask = labels_arr != -1

        test_mask  = (self._cached_context_ends_ts >= test_cutoff) & valid_mask
        val_mask   = (self._cached_context_ends_ts >= val_cutoff) & ~test_mask & valid_mask
        train_mask = (self._cached_context_ends_ts < val_cutoff) & valid_mask

        splits = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            idx = np.where(mask)[0]
            splits[name] = SplitDataset(
                X_tabular=self._cached_X_tab[idx],
                X_sequence=self._cached_contexts[idx],
                y=labels_arr[idx],
                feature_names=feat_names,
                metadata=[meta[i] for i in idx],
                split_name=name,
                label_fraction=label_fraction if name == "train" else 1.0,
            )
            logger.debug(splits[name].describe())

        return splits["train"], splits["val"], splits["test"]

    def build(
        self,
        merged_df: pd.DataFrame,
        terrain_df: pd.DataFrame,
        events_df: pd.DataFrame,
        label_fraction: float = 1.0,
        label_seed: int = 42,
    ) -> tuple["SplitDataset", "SplitDataset", "SplitDataset"]:
        """
        Build train / val / test splits.

        Args:
            merged_df: DataFrame with [zone_id, observed_at, + all time-series features].
                       Must be sorted by observed_at, UTC-aware.
            terrain_df: DataFrame with [zone_id, + static terrain features].
                        One row per zone.
            events_df: DataFrame with [zone_id, occurred_at, date_precision, ...].
            label_fraction: Fraction of positive labels to use (label-efficiency experiment).
            label_seed: Seed for label-efficiency subsampling.

        Returns:
            (train, val, test) SplitDataset tuples.
        """
        from ml.features.window_generator import WindowConfig, WindowGenerator
        from ml.features.label_builder import LabelBuilder, LabelConfig

        cfg = self.config
        test_cutoff = pd.Timestamp(cfg.test_cutoff, tz="UTC")
        val_cutoff = pd.Timestamp(cfg.val_cutoff, tz="UTC")

        # Generate windows
        win_cfg = WindowConfig(
            context_hours=cfg.context_hours,
            target_hours=cfg.target_hours,
            stride_hours=cfg.stride_hours,
            min_valid_fraction=cfg.min_valid_fraction,
        )
        gen = WindowGenerator(win_cfg)

        ts_feature_cols = [
            c for c in merged_df.columns
            if c not in ("zone_id", "observed_at", "data_source", "is_demo", "quality_flag")
        ]

        contexts_arr, _, meta = gen.generate_arrays(merged_df, include_targets=False)
        # contexts_arr: (N, T, F)
        logger.info(f"DatasetBuilder: generated {len(meta)} windows")

        if len(meta) == 0:
            logger.warning("DatasetBuilder: no windows generated — check input data length")
            empty = self._empty_split("train", ts_feature_cols)
            return empty, empty, empty

        # Build labels
        builder = LabelBuilder(LabelConfig(event_window_hours=cfg.target_hours))
        labels_df = builder.build(meta, events_df)

        # Apply label efficiency fraction
        if label_fraction < 1.0:
            labels_df = builder.apply_label_efficiency_fraction(
                labels_df, fraction=label_fraction, seed=label_seed
            )

        # Merge labels with metadata and contexts
        n = len(meta)
        for i, m in enumerate(meta):
            m["_idx"] = i
        labels_df["_idx"] = range(len(labels_df))

        # Build terrain lookup per zone
        terrain_lookup: dict[str, dict] = {}
        for _, row in terrain_df.iterrows():
            terrain_lookup[str(row["zone_id"])] = {
                feat: row.get(feat, np.nan) for feat in self.STATIC_FEATURES
            }

        # Extract per-window features
        X_tab_rows, X_seq_rows, y_rows, meta_rows = [], [], [], []

        for i, (m, lrow) in enumerate(zip(meta, labels_df.itertuples())):
            label = int(lrow.label)
            if label == -1:
                continue  # excluded (ambiguous negative or label-efficiency removed)

            ctx = contexts_arr[i]  # (T, F)

            # Snapshot features: last timestep of context window
            snapshot = {
                col: float(ctx[-1, j]) if j < ctx.shape[1] else np.nan
                for j, col in enumerate(ts_feature_cols)
                if col in self.SNAPSHOT_FEATURES
            }
            # Fill missing snapshot features
            for feat in self.SNAPSHOT_FEATURES:
                if feat not in snapshot:
                    snapshot[feat] = np.nan

            # Static terrain features for this zone (only if enabled)
            static_feats = self.STATIC_FEATURES if self.config.include_terrain else []
            if self.config.include_terrain:
                terrain_row = terrain_lookup.get(m["zone_id"], {})
                for feat in static_feats:
                    snapshot[feat] = terrain_row.get(feat, np.nan)

            # Build ordered feature vector
            all_feat_names = self.SNAPSHOT_FEATURES + static_feats
            x_tab = np.array([snapshot.get(f, np.nan) for f in all_feat_names], dtype=np.float32)

            X_tab_rows.append(x_tab)
            if self.config.include_sequence:
                X_seq_rows.append(ctx.astype(np.float32))
            y_rows.append(label)
            meta_rows.append(m)

        if not X_tab_rows:
            logger.warning("DatasetBuilder: no usable windows after label filtering")
            empty = self._empty_split("train", ts_feature_cols)
            return empty, empty, empty

        X_tab = np.stack(X_tab_rows)   # (N, F_tab)
        X_seq = np.stack(X_seq_rows) if self.config.include_sequence else None   # (N, T, F_seq)
        y = np.array(y_rows, dtype=np.int32)
        feat_names = self.SNAPSHOT_FEATURES + (self.STATIC_FEATURES if self.config.include_terrain else [])

        # ── Temporal split ────────────────────────────────────────────
        # Never shuffle — split by context_end timestamp
        context_ends = np.array([m["context_end"] for m in meta_rows])

        # Convert to comparable timestamps
        context_ends_ts = pd.to_datetime(context_ends, utc=True)
        test_mask  = context_ends_ts >= test_cutoff
        val_mask   = (context_ends_ts >= val_cutoff) & ~test_mask
        train_mask = ~test_mask & ~val_mask

        splits = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            idx = np.where(mask)[0]
            splits[name] = SplitDataset(
                X_tabular=X_tab[idx],
                X_sequence=X_seq[idx] if X_seq is not None else np.empty((len(idx), 0, 0), dtype=np.float32),
                y=y[idx],
                feature_names=feat_names,
                metadata=[meta_rows[i] for i in idx],
                split_name=name,
                label_fraction=label_fraction,
            )
            logger.info(splits[name].describe())

        return splits["train"], splits["val"], splits["test"]

    def _empty_split(self, name: str, ts_cols: list[str]) -> "SplitDataset":
        feat_names = self.SNAPSHOT_FEATURES + self.STATIC_FEATURES
        return SplitDataset(
            X_tabular=np.empty((0, len(feat_names)), dtype=np.float32),
            X_sequence=np.empty((0, self.config.context_hours, len(ts_cols)), dtype=np.float32),
            y=np.empty(0, dtype=np.int32),
            feature_names=feat_names,
            metadata=[],
            split_name=name,
        )
