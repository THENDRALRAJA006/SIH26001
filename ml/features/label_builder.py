"""
LAND-JEPA — Label Builder

Constructs binary landslide occurrence labels for windows.

CRITICAL RULES:
1. Labels are attached to the CONTEXT window based on whether a landslide
   occurs within the TARGET horizon after context_end.
2. date_precision MUST be respected:
   - 'exact': full precision, label the exact timestamp
   - 'day': label the full 24-hour day
   - 'month': EXCLUDED from precise labeling
   - 'year': EXCLUDED from precise labeling
   - 'unknown': EXCLUDED from precise labeling
3. A negative buffer is applied around known events to avoid
   ambiguous near-event negatives.
4. Labels must NEVER be constructed using information from after context_end.

Label uncertainty is documented in the output DataFrame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    """Configuration for label construction."""
    # Events within this many hours after context_end → label = 1
    event_window_hours: int = 24

    # Exclude windows from negatives if a known event falls within
    # this many hours BEFORE context_end (ambiguous pre-event period)
    negative_buffer_hours: int = 72

    # Which date_precision values are allowed for labeling
    # 'exact' and 'day' are usable. 'month'/'year'/'unknown' are excluded.
    allowed_precisions: tuple[str, ...] = ("exact", "day")


class LabelBuilder:
    """
    Assigns binary labels to window metadata records.

    Usage:
        builder = LabelBuilder(config)
        labels_df = builder.build(window_metadata, landslide_events_df)
    """

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def build(
        self,
        window_metadata: list[dict],
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign binary label (0/1) to each window.

        Args:
            window_metadata: List of dicts from WindowGenerator (zone_id, context_end, target_end).
            events_df: DataFrame with columns: zone_id, occurred_at, date_precision.

        Returns:
            DataFrame with columns: zone_id, context_end, target_end, label, label_confidence.
            label_confidence: 1.0 = high (exact precision), 0.8 = day, NaN if excluded.
        """
        if events_df is None or len(events_df) == 0:
            logger.warning("LabelBuilder: No landslide events provided. All labels will be 0.")
            return self._all_negative(window_metadata)

        # Filter to usable events
        usable_events = events_df[
            events_df["date_precision"].isin(self.config.allowed_precisions)
        ].copy()

        excluded_count = len(events_df) - len(usable_events)
        if excluded_count > 0:
            logger.info(
                f"LabelBuilder: Excluded {excluded_count} events with imprecise "
                f"date_precision (month/year/unknown). These will NOT be used as labels."
            )

        if len(usable_events) == 0:
            logger.warning("LabelBuilder: No usable events after precision filter. All labels will be 0.")
            return self._all_negative(window_metadata)

        # Ensure UTC
        usable_events = usable_events.copy()
        if usable_events["occurred_at"].dt.tz is None:
            usable_events["occurred_at"] = usable_events["occurred_at"].dt.tz_localize("UTC")

        records = []
        event_window_td = timedelta(hours=self.config.event_window_hours)
        negative_buffer_td = timedelta(hours=self.config.negative_buffer_hours)

        for meta in window_metadata:
            zone_id = meta["zone_id"]
            context_end = meta["context_end"]
            target_end = meta["target_end"]

            if context_end.tzinfo is None:
                context_end = context_end.replace(tzinfo=timezone.utc)
            if target_end.tzinfo is None:
                target_end = target_end.replace(tzinfo=timezone.utc)

            # Events for this zone
            zone_events = usable_events[usable_events["zone_id"] == zone_id]

            # Positive: any event within (context_end, context_end + event_window_hours]
            target_window_start = context_end
            target_window_end = context_end + event_window_td

            positive_events = zone_events[
                (zone_events["occurred_at"] > target_window_start) &
                (zone_events["occurred_at"] <= target_window_end)
            ]

            if len(positive_events) > 0:
                # Label = 1
                # Confidence based on precision of best event
                best_precision = positive_events["date_precision"].iloc[0]
                confidence = 1.0 if best_precision == "exact" else 0.8
                label = 1
            else:
                # Check for ambiguous negative (event within buffer BEFORE context_end)
                buffer_start = context_end - negative_buffer_td
                ambiguous_events = zone_events[
                    (zone_events["occurred_at"] >= buffer_start) &
                    (zone_events["occurred_at"] <= context_end)
                ]
                if len(ambiguous_events) > 0:
                    # Skip this window — too close to a known event to be a clean negative
                    label = -1   # -1 = excluded
                    confidence = float("nan")
                else:
                    label = 0
                    confidence = 1.0

            records.append({
                "zone_id": zone_id,
                "context_end": context_end,
                "target_start": meta["target_start"],
                "target_end": target_end,
                "label": label,
                "label_confidence": confidence,
                "context_start": meta.get("context_start"),
            })

        result = pd.DataFrame(records)

        # Statistics
        n_pos = (result["label"] == 1).sum()
        n_neg = (result["label"] == 0).sum()
        n_excl = (result["label"] == -1).sum()
        logger.info(
            f"LabelBuilder: {len(result)} windows — "
            f"positive={n_pos} ({n_pos/len(result):.1%}), "
            f"negative={n_neg} ({n_neg/len(result):.1%}), "
            f"excluded={n_excl} ({n_excl/len(result):.1%})"
        )

        if n_pos == 0:
            logger.warning(
                "LabelBuilder: No positive labels found. "
                "Check that event timestamps align with the window time range."
            )

        return result

    def _all_negative(self, window_metadata: list[dict]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "zone_id": m["zone_id"],
                "context_end": m["context_end"],
                "target_start": m.get("target_start"),
                "target_end": m.get("target_end"),
                "label": 0,
                "label_confidence": 1.0,
                "context_start": m.get("context_start"),
            }
            for m in window_metadata
        ])

    def apply_label_efficiency_fraction(
        self,
        labels_df: pd.DataFrame,
        fraction: float,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Subsample the positive labels to simulate label-efficient training.

        This implements the label-efficiency experiment:
        - fraction=1.0: all labels available
        - fraction=0.05: only 5% of positive labels visible

        Negatives are NOT subsampled (we know regions without events are negatives).
        Only positives are made "invisible" by converting them to excluded (-1).

        Args:
            labels_df: Output of build().
            fraction: Fraction of positive labels to KEEP (0.05 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Modified labels_df with some positives converted to excluded.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        result = labels_df.copy()
        positive_mask = result["label"] == 1
        positive_idx = result.index[positive_mask].tolist()

        if not positive_idx:
            return result

        rng = np.random.default_rng(seed)
        n_keep = max(1, int(len(positive_idx) * fraction))
        keep_idx = set(rng.choice(positive_idx, size=n_keep, replace=False).tolist())
        remove_idx = [i for i in positive_idx if i not in keep_idx]

        # Convert removed positives to excluded
        result.loc[remove_idx, "label"] = -1
        result.loc[remove_idx, "label_confidence"] = float("nan")

        logger.info(
            f"LabelBuilder.apply_fraction: fraction={fraction:.0%}, "
            f"seed={seed}, kept={n_keep}/{len(positive_idx)} positives"
        )
        return result
