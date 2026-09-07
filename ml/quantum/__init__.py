"""
LAND-JEPA — Quantum Module (EXPERIMENTAL RESEARCH BRANCH)
==========================================================
SIH26001 / Team ZAIX

PURPOSE:
  Research question: Can a Variational Quantum Classifier (VQC) classify
  compact LAND-JEPA representations competitively with a matched classical
  classifier?

STATUS: EXPERIMENTAL — NOT USED FOR EMERGENCY ALERTS

Production path (unchanged):
  LAND-JEPA → classical risk head → GIS + alerts

Research path (this module):
  LAND-JEPA embedding → PCA → VQC → research comparison only

IMPORTANT:
  - Do NOT import this module in production alert code paths.
  - All results must be clearly labelled "QUANTUM SIMULATION".
  - No quantum advantage claim is made unless rigourously demonstrated.

Requires: pennylane >= 0.40, scikit-learn, numpy, torch
"""

from ml.quantum.quantum_features import (
    LandJEPAEmbeddingExtractor,
    QuantumFeatureReducer,
    QuantumFeatureScaler,
)
from ml.quantum.vqc_circuit import build_vqc_circuit, VQCCircuitConfig
from ml.quantum.vqc_model import VQCClassifier, ClassicalMatchedBaseline
from ml.quantum.vqc_trainer import VQCTrainer
from ml.quantum.vqc_evaluator import VQCEvaluator

__all__ = [
    "LandJEPAEmbeddingExtractor",
    "QuantumFeatureReducer",
    "QuantumFeatureScaler",
    "build_vqc_circuit",
    "VQCCircuitConfig",
    "VQCClassifier",
    "ClassicalMatchedBaseline",
    "VQCTrainer",
    "VQCEvaluator",
]
