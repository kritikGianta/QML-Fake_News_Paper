from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    data_csv: Path
    outputs_dir: Path

    random_seed: int = 42

    test_size: float = 0.25

    # Quantum settings
    pegasos_lambda: float = 0.02
    pegasos_iterations: int = 200

    # PCA / qubits sweep
    pca_components_grid: tuple[int, ...] = (2, 3, 4)
    qubits_grid: tuple[int, ...] = (2, 3, 4)

    # Noise sweep (depolarizing probability)
    noise_probs: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02)
