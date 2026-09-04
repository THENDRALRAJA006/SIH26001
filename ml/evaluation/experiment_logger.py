"""
LAND-JEPA — Experiment Logger

Records all experiment runs to a JSON Lines file for reproducibility.
Each line is one experiment result (model × split × label_fraction × seed).

Design:
  - Append-only (never overwrites past results).
  - Human-readable JSON Lines format.
  - No external dependencies (MLflow optional).
  - Each entry includes git commit hash for reproducibility.

Usage:
    logger = ExperimentLogger("results/experiment_log.jsonl")
    logger.log(result, config_snapshot, git_commit="abc123")
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_git_commit() -> str | None:
    """Return short git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


class ExperimentLogger:
    """
    Append-only experiment logger writing JSON Lines.

    Every logged entry has:
      - timestamp: ISO8601 UTC
      - git_commit: short hash for reproducibility
      - model_name, split, metrics, config_snapshot
      - label_fraction, seed
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExperimentLogger: writing to {self.log_path}")

    def log(
        self,
        metrics: dict[str, Any],
        config_snapshot: dict[str, Any] | None = None,
        model_name: str = "",
        git_commit: str | None = None,
        notes: str = "",
    ) -> None:
        """
        Append one experiment result to the log file.

        Args:
            metrics: Dict from EvaluationResult.as_dict().
            config_snapshot: Hyperparameter config used for this run.
            model_name: Model identifier.
            git_commit: Git commit hash. Auto-detected if None.
            notes: Optional freeform notes.
        """
        if git_commit is None:
            git_commit = _get_git_commit()

        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "git_commit": git_commit,
            "model_name": model_name or metrics.get("model_name", "unknown"),
            "metrics": metrics,
            "config_snapshot": config_snapshot or {},
            "notes": notes,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.info(
            f"ExperimentLogger: logged {model_name or 'run'} "
            f"AUCPR={metrics.get('aucpr', 'N/A')}"
        )

    def load_all(self) -> list[dict]:
        """Load all experiment entries from the log file."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed log line: {e}")
        return entries

    def summarize(self) -> None:
        """Print a summary table of all logged experiments."""
        entries = self.load_all()
        if not entries:
            print("No experiments logged yet.")
            return

        print(f"\n{'='*80}")
        print(f"Experiment Log: {self.log_path}")
        print(f"{'='*80}")
        header = f"{'Model':<25} {'Split':<6} {'LblFrac':>8} {'AUCPR':>7} {'AUROC':>7} {'F1':>7} {'P':>6} {'R':>6}"
        print(header)
        print("-" * 80)
        for e in entries:
            m = e.get("metrics", {})
            print(
                f"{e.get('model_name', 'N/A'):<25} "
                f"{m.get('split', ''):<6} "
                f"{m.get('label_fraction', 1.0):>8.0%} "
                f"{m.get('aucpr', 0):>7.4f} "
                f"{m.get('auroc', 0):>7.4f} "
                f"{m.get('f1', 0):>7.4f} "
                f"{m.get('precision', 0):>6.3f} "
                f"{m.get('recall', 0):>6.3f}"
            )
        print(f"{'='*80}\n")
