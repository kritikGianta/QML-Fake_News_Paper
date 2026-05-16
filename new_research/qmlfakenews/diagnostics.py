from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def center_gram(K: NDArray[np.floating]) -> NDArray[np.floating]:
    n = K.shape[0]
    one = np.ones((n, n), dtype=float) / n
    return K - one @ K - K @ one + one @ K @ one


def kernel_target_alignment(K: NDArray[np.floating], y: NDArray[np.integer], *, centered: bool = True) -> float:
    y = y.astype(float)
    Y = np.outer(y, y)

    K_use = center_gram(K) if centered else K
    Y_use = center_gram(Y) if centered else Y

    num = float(np.sum(K_use * Y_use))
    den = float(np.linalg.norm(K_use, ord="fro") * np.linalg.norm(Y_use, ord="fro"))
    return num / den if den > 0 else float("nan")


@dataclass(frozen=True)
class SpectrumStats:
    eigenvalues: NDArray[np.floating]
    spectral_gap: float
    condition_number: float
    effective_rank: float
    leading_eigen_ratio: float


def spectrum_stats(K: NDArray[np.floating], *, centered: bool = True, ridge: float = 1e-8) -> SpectrumStats:
    K_use = center_gram(K) if centered else K

    # Ensure symmetric
    K_sym = 0.5 * (K_use + K_use.T)

    # Ridge for numerical stability
    K_sym = K_sym + ridge * np.eye(K_sym.shape[0])

    evals = np.linalg.eigvalsh(K_sym)
    evals_sorted = np.sort(evals)[::-1]

    if evals_sorted.size >= 2:
        gap = float(evals_sorted[0] - evals_sorted[1])
    else:
        gap = float("nan")

    # Condition number: max / min positive
    positive = evals_sorted[evals_sorted > 0]
    if positive.size == 0:
        cond = float("inf")
    else:
        cond = float(positive.max() / positive.min())

    # Effective rank (a.k.a. participation ratio): (sum λ)^2 / sum λ^2 over positive λ.
    # This measures how many eigen-directions meaningfully contribute.
    if positive.size == 0:
        eff_rank = float("nan")
        lead_ratio = float("nan")
    else:
        s1 = float(np.sum(positive))
        s2 = float(np.sum(positive**2))
        eff_rank = (s1 * s1 / s2) if s2 > 0 else float("nan")
        lead_ratio = float(positive.max() / s1) if s1 > 0 else float("nan")

    return SpectrumStats(
        eigenvalues=evals_sorted,
        spectral_gap=gap,
        condition_number=cond,
        effective_rank=eff_rank,
        leading_eigen_ratio=lead_ratio,
    )
