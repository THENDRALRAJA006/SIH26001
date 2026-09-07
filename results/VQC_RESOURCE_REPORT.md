# VQC Resource Report

**SIH26001 / Team ZAIX**

> [!IMPORTANT]
> EXPERIMENTAL RESEARCH BRANCH — QUANTUM SIMULATION ONLY
> NOT CONNECTED TO REAL QUANTUM HARDWARE
> NOT USED FOR EMERGENCY ALERTS

---

## Simulator

| Property | Value |
|---|---|
| Backend | PennyLane `default.qubit` |
| Mode | Statevector simulation (analytic) |
| Shots | None (analytic gradients) |
| Interface | PennyLane autograd |
| Real hardware | NOT CONNECTED |

---

## Circuit Configurations Evaluated

| Config | Qubits | Depth | Encoding | Entanglement | Trainable Params |
|---|---|---|---|---|---|
| VQC-4q-d2 | 4 | 2 | Angle (RY) | CNOT chain | 16 |
| VQC-4q-d4 | 4 | 4 | Angle (RY) | CNOT chain | 32 |
| VQC-4q-d6 | 4 | 6 | Angle (RY) | CNOT chain | 48 |
| VQC-8q-d2 | 8 | 2 | Angle (RY) | CNOT chain | 32 |
| VQC-8q-d4 | 8 | 4 | Angle (RY) | CNOT chain | 64 |
| VQC-8q-d6 | 8 | 6 | Angle (RY) | CNOT chain | 96 |

---

## Mathematical Encoding

```
Input x ∈ [0, π]^n  (after PCA + MinMaxScaler)

Qubit i:  RY(x[i]) |0⟩

Variational layer (repeated depth times):
  RY(θ[l,i]) per qubit i
  RZ(φ[l,i]) per qubit i
  CNOT(i, i+1) chain

Measurement:  ⟨ψ| Z_0 |ψ⟩  →  σ(E)  =  P(landslide=1)
```

---

## Training Seeds
Seeds: [42, 123, 456]

## Label Fractions
1%, 5%, 10%, 25%, 50%, 100%

---

## Classical Matched Baselines

Use EXACTLY the same PCA-reduced features as VQC:

- **Logistic Regression**: sklearn, C=1.0, class_weight=balanced
- **MLP**: PyTorch 2-layer [32, 16], ReLU, Adam, BCE loss

---

## Safety

> [!CAUTION]
> VQC results are SIMULATOR ONLY.
> They are NOT validated on real quantum hardware.
> They are NOT used in the production emergency alert path.
> The production LAND-JEPA classical risk head remains unchanged.