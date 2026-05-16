from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .feature_maps import FeatureMapName, build_feature_map


@dataclass
class QuantumKernelFactory:
    feature_map_name: FeatureMapName
    num_qubits: int
    reps: int = 2
    seed: int = 42
    shots: Optional[int] = None
    noise_prob: float = 0.0

    def build(self):
        """Creates a FidelityQuantumKernel configured for either ideal or noisy simulation."""
        from qiskit_algorithms.utils import algorithm_globals

        algorithm_globals.random_seed = self.seed

        feature_map = build_feature_map(self.feature_map_name, self.num_qubits, reps=int(self.reps))

        # Qiskit ML (this version) accepts a `fidelity` object.
        # For noise robustness we build a ComputeUncompute fidelity powered by a noisy Aer Sampler.
        if self.noise_prob and self.noise_prob > 0:
            sampler, transpiler = _build_noisy_sampler(noise_prob=self.noise_prob, shots=self.shots, seed=self.seed)
            from qiskit_algorithms.state_fidelities import ComputeUncompute
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            fidelity = ComputeUncompute(sampler=sampler, shots=self.shots, transpiler=transpiler)
            return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

        # Ideal/statevector path (fast for small qubits)
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        return FidelityQuantumKernel(feature_map=feature_map)


def _build_noisy_sampler(*, noise_prob: float, shots: Optional[int], seed: int):
    """Build a sampler backed by AerSimulator with depolarizing noise.

    This function is written to be resilient across Qiskit versions; if a specific
    sampler class isn't available, it will raise a helpful error.
    """

    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer import AerSimulator
    from qiskit.primitives import BackendSamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    noise_model = NoiseModel()

    error_1q = depolarizing_error(noise_prob, 1)
    error_2q = depolarizing_error(min(1.0, 2 * noise_prob), 2)

    one_qubit_gates = ["rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
    two_qubit_gates = ["cx", "cz", "swap"]

    for g in one_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_1q, g)
    for g in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_2q, g)

    # Construct a V2 sampler required by ComputeUncompute.
    # We configure noise on the backend itself.
    backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    options = {"seed_simulator": int(seed)}
    if shots is not None:
        options["default_shots"] = int(shots)

    sampler = BackendSamplerV2(backend=backend, options=options)
    transpiler = generate_preset_pass_manager(backend=backend, optimization_level=1)
    return sampler, transpiler


def evaluate_kernel_matrix(kernel, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
    return np.asarray(kernel.evaluate(x_vec=X_left, y_vec=X_right), dtype=float)
