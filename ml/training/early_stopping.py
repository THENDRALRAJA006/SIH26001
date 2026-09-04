"""
LAND-JEPA — Early Stopping Callback

Monitors a validation metric and stops training when it stops improving.
Saves the best model state dict for restoration after stopping.

Monitors AUCPR (higher is better) by default.
"""
from __future__ import annotations

import copy
import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

Mode = Literal["max", "min"]


class EarlyStopping:
    """
    Early stopping with best-model restoration.

    Usage:
        es = EarlyStopping(patience=10, mode="max")
        for epoch in range(max_epochs):
            val_aucpr = evaluate(...)
            if es.step(val_aucpr, model):
                break
        es.restore(model)  # restores best weights

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: 'max' (AUCPR, F1) or 'min' (loss).
        min_delta: Minimum improvement to qualify as an improvement.
        restore_best: If True, restores best weights when stopped.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: Mode = "max",
        min_delta: float = 1e-4,
        restore_best: bool = True,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best_score: float = float("-inf") if mode == "max" else float("inf")
        self._best_state: dict | None = None
        self._wait: int = 0
        self._stopped_epoch: int = 0
        self._best_epoch: int = 0

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Process one validation score.

        Returns:
            True if training should stop.
        """
        improved = (
            score > self._best_score + self.min_delta
            if self.mode == "max"
            else score < self._best_score - self.min_delta
        )

        if improved:
            self._best_score = score
            self._best_state = copy.deepcopy(model.state_dict())
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            logger.info(
                f"EarlyStopping: no improvement for {self.patience} epochs. "
                f"Best {self.mode}={self._best_score:.4f}"
            )
            return True  # stop

        return False  # continue

    def restore(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self._best_state is not None and self.restore_best:
            model.load_state_dict(self._best_state)
            logger.info(
                f"EarlyStopping: restored best model "
                f"(best_score={self._best_score:.4f})"
            )
        else:
            logger.warning("EarlyStopping: no saved state to restore.")

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def wait(self) -> int:
        return self._wait
