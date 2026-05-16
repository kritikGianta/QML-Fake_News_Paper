from __future__ import annotations

from typing import Literal


FeatureMapName = Literal["z", "zz", "pauli_xyz", "ry_custom"]


def build_feature_map(name: FeatureMapName, num_qubits: int, *, reps: int = 2):
    """Builds a Qiskit feature map circuit.

    Import is inside the function so the rest of the pipeline works even when
    Qiskit isn't installed (e.g., running only classical baselines).
    """

    if name == "z":
        from qiskit.circuit.library import ZFeatureMap

        return ZFeatureMap(feature_dimension=num_qubits, reps=int(reps))

    if name == "zz":
        from qiskit.circuit.library import ZZFeatureMap

        return ZZFeatureMap(feature_dimension=num_qubits, reps=int(reps), entanglement="linear")

    if name == "pauli_xyz":
        from qiskit.circuit.library import PauliFeatureMap

        return PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=int(reps),
            paulis=["X", "Y", "Z"],
            entanglement="full",
        )

    if name == "ry_custom":
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        theta = ParameterVector("theta", num_qubits)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.ry(theta[i], i)
        return qc

    raise ValueError(f"Unknown feature map: {name}")
