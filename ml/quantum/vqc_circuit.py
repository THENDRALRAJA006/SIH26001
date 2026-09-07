"""
LAND-JEPA — VQC Circuit Definition (EXPERIMENTAL)
===================================================
SIH26001 / Team ZAIX

EXPERIMENTAL RESEARCH BRANCH — QUANTUM SIMULATION ONLY.
NOT USED FOR EMERGENCY ALERTS.

Circuit architecture:
  1. Angle Encoding:   RY(x_i) per qubit i  (x ∈ [0, π])
  2. Variational block (repeated `depth` times):
       a. Trainable RY(θ_i) per qubit
       b. Trainable RZ(φ_i) per qubit
       c. CNOT chain: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1)
  3. Measurement: <PauliZ on qubit 0> → mapped to probability via sigmoid

Mathematical formulation:
  For n qubits, depth d, features x ∈ [0,π]^n:

  |ψ⟩ = U_var(θ^d, φ^d) · CNOT_chain · ... · U_var(θ^1, φ^1) · CNOT_chain
         · ⊗_i RY(x_i) |0⟩^⊗n

  Output = σ(<ψ| Z_0 |ψ>)  where σ is sigmoid

  Trainable parameters: 2 × n × depth  (n RY + n RZ per layer)

Simulator: PennyLane default.qubit (statevector, analytic mode)
Hardware: NOT CONNECTED — simulation only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pennylane as qml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VQCCircuitConfig:
    """
    Complete specification of a VQC circuit.

    Attributes:
        n_qubits:    Number of qubits. Must equal n_pca_components.
        depth:       Number of variational layers.
        encoding:    Feature encoding method ('angle').
        device_name: PennyLane device identifier.
        shots:       None = analytic (statevector); int = shot-based.
        seed:        Random seed for reproducibility.

    Trainable parameters: 2 × n_qubits × depth
    """
    n_qubits: int = 4
    depth: int = 2
    encoding: str = "angle"     # RY encoding
    device_name: str = "default.qubit"
    shots: int | None = None    # None = analytic
    seed: int = 42

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return 2 * self.n_qubits * self.depth

    @property
    def circuit_label(self) -> str:
        return f"VQC-{self.n_qubits}q-d{self.depth}"

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "encoding": self.encoding,
            "device_name": self.device_name,
            "shots": self.shots,
            "seed": self.seed,
            "n_params": self.n_params,
            "circuit_label": self.circuit_label,
            "entanglement": "cnot_chain",
            "measurement": "PauliZ_qubit0",
            "output": "sigmoid(expectation_value)",
            "simulation_mode": "QUANTUM SIMULATION",
            "hardware": "NOT CONNECTED",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_vqc_circuit(config: VQCCircuitConfig) -> Callable:
    """
    Build and return a PennyLane QNode implementing the VQC circuit.

    Args:
        config: VQCCircuitConfig specifying the circuit architecture.

    Returns:
        qnode: A callable QNode that takes (params, x) and returns
               the expectation value <Z_0>.

    Circuit structure per call:
        - Phase 1: Angle encoding  — RY(x[i]) for i in range(n_qubits)
        - Phase 2: Repeat depth times:
            a. Trainable layer — RY(θ[l,i]), RZ(φ[l,i]) per qubit
            b. Entanglement   — CNOT(i, i+1) chain
        - Phase 3: Measurement — qml.expval(qml.PauliZ(0))
    """
    n_qubits = config.n_qubits
    depth = config.depth

    dev = qml.device(config.device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params: np.ndarray, x: np.ndarray) -> float:
        """
        Args:
            params: (2, depth, n_qubits) array of trainable parameters.
                    params[0] = RY angles, params[1] = RZ angles.
            x:      (n_qubits,) input feature vector in [0, π].

        Returns:
            Expectation value of PauliZ on qubit 0.
        """
        # ── Phase 1: Angle Encoding ─────────────────────────────────────────
        # Each feature maps to an RY rotation: RY(x_i)|0⟩ on qubit i
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        # ── Phase 2: Variational Layers ─────────────────────────────────────
        for layer in range(depth):
            # Trainable RY gates
            for i in range(n_qubits):
                qml.RY(params[0, layer, i], wires=i)
            # Trainable RZ gates
            for i in range(n_qubits):
                qml.RZ(params[1, layer, i], wires=i)
            # CNOT entanglement chain: qubit 0 → 1 → 2 → ... → n-1
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # ── Phase 3: Measurement ────────────────────────────────────────────
        return qml.expval(qml.PauliZ(0))

    logger.info(
        f"[VQC] Built circuit: {config.circuit_label} | "
        f"params={config.n_params} | device={config.device_name} | "
        f"shots={config.shots} | MODE=QUANTUM SIMULATION"
    )
    return circuit


def initialise_params(config: VQCCircuitConfig, seed: int | None = None) -> np.ndarray:
    """
    Initialise variational circuit parameters.

    Shape: (2, depth, n_qubits)
      dim 0: gate type (0=RY, 1=RZ)
      dim 1: layer index
      dim 2: qubit index

    Initialisation: uniform in [−π, π] with controlled seed.
    """
    rng = np.random.default_rng(seed if seed is not None else config.seed)
    params = rng.uniform(-np.pi, np.pi, size=(2, config.depth, config.n_qubits))
    return params.astype(np.float64)  # PennyLane autograd requires float64


def circuit_to_probability(expectation_value: float) -> float:
    """
    Convert circuit expectation value <Z_0> ∈ [-1, +1] to a probability.

    Mapping: prob = σ(expectation_value) = 1 / (1 + exp(-E))
    where E = expectation_value.

    This is numerically stable for all E ∈ [-1, +1].
    """
    # Sigmoid of the expectation value
    return float(1.0 / (1.0 + np.exp(-expectation_value)))


def print_circuit_info(config: VQCCircuitConfig) -> None:
    """Log a human-readable circuit resource summary."""
    logger.info("=" * 60)
    logger.info(f"  VQC CIRCUIT RESOURCE REPORT ({config.circuit_label})")
    logger.info("=" * 60)
    logger.info(f"  Mode:           QUANTUM SIMULATION")
    logger.info(f"  Device:         {config.device_name}")
    logger.info(f"  Qubits:         {config.n_qubits}")
    logger.info(f"  Circuit depth:  {config.depth} variational layers")
    logger.info(f"  Encoding:       Angle (RY per qubit)")
    logger.info(f"  Entanglement:   CNOT chain")
    logger.info(f"  Trainable gates: RY + RZ per qubit per layer")
    logger.info(f"  Total params:   {config.n_params}")
    logger.info(f"  Shots:          {'Analytic' if config.shots is None else config.shots}")
    logger.info(f"  Measurement:    <PauliZ on qubit 0>")
    logger.info(f"  Output:         sigmoid(expectation_value)")
    logger.info(f"  NOT for production alerts.")
    logger.info("=" * 60)
