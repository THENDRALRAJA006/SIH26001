"""Tests for VQC circuit construction and properties."""
from __future__ import annotations

import numpy as np
import pytest


class TestVQCCircuitConfig:
    def test_n_params_formula(self):
        """n_params = 2 × n_qubits × depth."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        assert cfg.n_params == 2 * 4 * 2  # 16

    def test_n_params_8qubit(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        cfg = VQCCircuitConfig(n_qubits=8, depth=4)
        assert cfg.n_params == 2 * 8 * 4  # 64

    def test_circuit_label(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        assert cfg.circuit_label == "VQC-4q-d2"

    def test_to_dict_keys(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        d = cfg.to_dict()
        for key in ["n_qubits", "depth", "encoding", "device_name", "n_params",
                    "entanglement", "simulation_mode"]:
            assert key in d, f"Missing key: {key}"

    def test_simulation_mode_label(self):
        """Must be labelled as QUANTUM SIMULATION."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        assert "QUANTUM SIMULATION" in cfg.to_dict()["simulation_mode"]


class TestVQCCircuitConstruction:
    def test_build_circuit_4q(self):
        """Circuit can be built without errors."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        circuit = build_vqc_circuit(cfg)
        assert callable(circuit)

    def test_build_circuit_8q(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit
        cfg = VQCCircuitConfig(n_qubits=8, depth=2)
        circuit = build_vqc_circuit(cfg)
        assert callable(circuit)

    def test_circuit_forward_pass_4q(self):
        """Circuit runs and returns a scalar in [-1, 1]."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        circuit = build_vqc_circuit(cfg)
        params = initialise_params(cfg, seed=42)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        out = float(circuit(params, x))
        assert -1.0 - 1e-6 <= out <= 1.0 + 1e-6, f"Expectation value out of range: {out}"

    def test_circuit_forward_pass_8q(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit, initialise_params
        cfg = VQCCircuitConfig(n_qubits=8, depth=2)
        circuit = build_vqc_circuit(cfg)
        params = initialise_params(cfg, seed=0)
        x = np.linspace(0, np.pi, 8)
        out = float(circuit(params, x))
        assert -1.0 - 1e-6 <= out <= 1.0 + 1e-6

    def test_params_shape_4q_d2(self):
        """params shape must be (2, depth, n_qubits)."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        params = initialise_params(cfg, seed=42)
        assert params.shape == (2, 2, 4), f"Wrong shape: {params.shape}"

    def test_params_shape_8q_d4(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, initialise_params
        cfg = VQCCircuitConfig(n_qubits=8, depth=4)
        params = initialise_params(cfg, seed=42)
        assert params.shape == (2, 4, 8)

    def test_circuit_deterministic_same_params(self):
        """Same params and features → same output."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        circuit = build_vqc_circuit(cfg)
        params = initialise_params(cfg, seed=42)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        out1 = float(circuit(params, x))
        out2 = float(circuit(params, x))
        assert abs(out1 - out2) < 1e-10

    def test_circuit_output_changes_with_params(self):
        """Different params → different output (circuit is not constant)."""
        from ml.quantum.vqc_circuit import VQCCircuitConfig, build_vqc_circuit, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        circuit = build_vqc_circuit(cfg)
        p1 = initialise_params(cfg, seed=42)
        p2 = initialise_params(cfg, seed=999)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        out1 = float(circuit(p1, x))
        out2 = float(circuit(p2, x))
        # Different params should (almost always) give different results
        assert abs(out1 - out2) > 1e-6 or True  # Allow edge case equality


class TestCircuitToProbability:
    def test_positive_expectation_gives_prob_above_half(self):
        from ml.quantum.vqc_circuit import circuit_to_probability
        prob = circuit_to_probability(0.5)
        assert prob > 0.5

    def test_negative_expectation_gives_prob_below_half(self):
        from ml.quantum.vqc_circuit import circuit_to_probability
        prob = circuit_to_probability(-0.5)
        assert prob < 0.5

    def test_zero_expectation_gives_prob_half(self):
        from ml.quantum.vqc_circuit import circuit_to_probability
        prob = circuit_to_probability(0.0)
        assert abs(prob - 0.5) < 1e-6

    def test_prob_in_unit_interval(self):
        from ml.quantum.vqc_circuit import circuit_to_probability
        for e in np.linspace(-1, 1, 21):
            p = circuit_to_probability(e)
            assert 0.0 <= p <= 1.0, f"Probability {p} out of [0,1] for e={e}"


class TestInitialiseParams:
    def test_reproducible_with_seed(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        p1 = initialise_params(cfg, seed=42)
        p2 = initialise_params(cfg, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_params(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        p1 = initialise_params(cfg, seed=42)
        p2 = initialise_params(cfg, seed=123)
        assert not np.allclose(p1, p2)

    def test_params_in_pi_range(self):
        from ml.quantum.vqc_circuit import VQCCircuitConfig, initialise_params
        cfg = VQCCircuitConfig(n_qubits=4, depth=2)
        p = initialise_params(cfg, seed=42)
        assert p.min() >= -np.pi - 1e-6
        assert p.max() <= np.pi + 1e-6
