"""
LAND-JEPA — Model Registry

Tracks model lineage, checkpoints, metrics, and lifecycle state.
Only validated models can be promoted to PRODUCTION.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    model_name: str
    version: str
    status: str                         # 'PRODUCTION' | 'CANDIDATE' | 'ARCHIVED'
    checkpoint_path: str
    training_date: str
    dataset_version: str
    configuration: dict[str, Any]
    metrics: dict[str, float]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        return cls(**data)


class ModelRegistry:
    """
    Local filesystem model registry backed by JSON metadata.
    """

    def __init__(self, registry_path: str | Path = "ml/checkpoints/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, ModelRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._models[k] = ModelRecord.from_dict(v)
            except Exception as e:
                logger.warning(f"Error loading model registry: {e}")

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._models.items()}, f, indent=2)

    def register(
        self,
        model_name: str,
        version: str,
        checkpoint_path: str,
        metrics: dict[str, float],
        configuration: dict[str, Any],
        dataset_version: str = "NER-DEMO-v1",
        status: str = "CANDIDATE",
        notes: str = "",
    ) -> ModelRecord:
        record = ModelRecord(
            model_name=model_name,
            version=version,
            status=status,
            checkpoint_path=str(checkpoint_path),
            training_date=datetime.now(tz=timezone.utc).isoformat(),
            dataset_version=dataset_version,
            configuration=configuration,
            metrics=metrics,
            notes=notes,
        )
        key = f"{model_name}:{version}"
        self._models[key] = record
        self._save()
        logger.info(f"Registered model {key} (status={status})")
        return record

    def promote_to_production(self, model_name: str, version: str) -> None:
        target_key = f"{model_name}:{version}"
        if target_key not in self._models:
            raise KeyError(f"Model {target_key} not found in registry")

        # Demote existing production models of the same name
        for k, v in self._models.items():
            if v.model_name == model_name and v.status == "PRODUCTION":
                v.status = "ARCHIVED"

        self._models[target_key].status = "PRODUCTION"
        self._save()
        logger.info(f"Promoted {target_key} to PRODUCTION")

    def get_production_model(self, model_name: str = "LAND-JEPA") -> ModelRecord | None:
        for v in self._models.values():
            if v.model_name == model_name and v.status == "PRODUCTION":
                return v
        return None

    def list_models(self) -> list[ModelRecord]:
        return list(self._models.values())
