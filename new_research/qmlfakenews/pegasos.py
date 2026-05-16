from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PegasosHistory:
    iterations: list[int]
    n_support: list[int]
    train_accuracy: list[float]
    test_accuracy: list[float]


class KernelPegasosSVM:
    """Kernelized Pegasos for binary labels in {-1, +1}.

    This implements the standard kernel Pegasos update:
    - sample i
    - f(x_i) = (1/(lambda*t)) * sum_j alpha_j y_j K(x_j, x_i)
    - if y_i f(x_i) < 1: alpha_i += 1

    Prediction uses the final scaling 1/(lambda*T).

    Notes:
    - We store counts alpha_j (integers) for each stored support vector.
    - This is still an online method; convergence is empirical.
    """

    def __init__(
        self,
        kernel,
        *,
        lambda_reg: float,
        iterations: int,
        seed: int = 42,
    ):
        self.kernel = kernel
        self.lambda_reg = float(lambda_reg)
        self.iterations = int(iterations)
        self.seed = int(seed)

        self.support_x: List[np.ndarray] = []
        self.support_y: List[int] = []
        self.alpha_counts: List[int] = []
        self.support_indices: List[int] = []
        self._index_to_pos: Dict[int, int] = {}
        self._T: int = 0

        # Training-time cache for kernel values between training indices and support-vector indices.
        # Keyed by (train_index, support_train_index).
        self._train_kernel_cache: Dict[tuple[int, int], float] = {}

    @property
    def n_support_vectors(self) -> int:
        return len(self.support_x)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        eval_every: int = 20,
        eval_subset_n: int = 250,
    ) -> PegasosHistory:
        rng = np.random.default_rng(self.seed)

        self.support_x = []
        self.support_y = []
        self.alpha_counts = []
        self.support_indices = []
        self._index_to_pos = {}
        self._T = 0
        self._train_kernel_cache = {}

        history = PegasosHistory(iterations=[], n_support=[], train_accuracy=[], test_accuracy=[])

        # To keep training lightweight (especially for quantum kernels), evaluate accuracy on subsets.
        eval_subset_n = int(eval_subset_n)
        train_eval_idx = None
        test_eval_idx = None
        if eval_every and eval_subset_n > 0:
            n_train = int(X_train.shape[0])
            n_eval_tr = min(eval_subset_n, n_train)
            train_eval_idx = rng.choice(n_train, size=n_eval_tr, replace=False)

            if X_test is not None and y_test is not None:
                n_test = int(X_test.shape[0])
                n_eval_te = min(eval_subset_n, n_test)
                test_eval_idx = rng.choice(n_test, size=n_eval_te, replace=False)

        n = X_train.shape[0]
        for t in range(1, self.iterations + 1):
            self._T = t
            i = int(rng.integers(0, n))

            x_i = X_train[i]
            y_i = int(y_train[i])

            margin = y_i * self._decision_for_train_index(i, X_train)
            if margin < 1.0:
                if i in self._index_to_pos:
                    pos = self._index_to_pos[i]
                    self.alpha_counts[pos] += 1
                else:
                    pos = len(self.support_x)
                    self._index_to_pos[i] = pos
                    self.support_indices.append(i)
                    self.support_x.append(x_i)
                    self.support_y.append(y_i)
                    self.alpha_counts.append(1)

            if eval_every and (t % eval_every == 0 or t == 1 or t == self.iterations):
                history.iterations.append(t)
                history.n_support.append(self.n_support_vectors)

                if train_eval_idx is not None:
                    history.train_accuracy.append(float(np.mean(self.predict(X_train[train_eval_idx]) == y_train[train_eval_idx])))
                else:
                    history.train_accuracy.append(float(np.mean(self.predict(X_train) == y_train)))

                if X_test is not None and y_test is not None:
                    if test_eval_idx is not None:
                        history.test_accuracy.append(float(np.mean(self.predict(X_test[test_eval_idx]) == y_test[test_eval_idx])))
                    else:
                        history.test_accuracy.append(float(np.mean(self.predict(X_test) == y_test)))
                else:
                    history.test_accuracy.append(float("nan"))

        return history

    def _decision_for_train_index(self, i: int, X_train: np.ndarray) -> float:
        """Fast decision score for a training sample index using a cache.

        This avoids repeatedly invoking expensive kernel evaluations for the same
        (training-point, support-vector) pairs during online updates.
        """

        if not self.support_x:
            return 0.0

        T = max(1, self._T)
        scale = 1.0 / (self.lambda_reg * T)

        # Determine which support vectors are missing from the cache.
        support_train_idx = self.support_indices
        missing_positions: list[int] = []
        for pos, j in enumerate(support_train_idx):
            if (i, j) not in self._train_kernel_cache:
                missing_positions.append(pos)

        if missing_positions:
            y_vec = np.asarray([self.support_x[pos] for pos in missing_positions])
            k_vals = self.kernel.evaluate(x_vec=np.asarray([X_train[i]]), y_vec=y_vec)
            k_vals = np.asarray(k_vals, dtype=float).reshape(-1)
            for k, pos in zip(k_vals, missing_positions):
                j = support_train_idx[pos]
                self._train_kernel_cache[(i, j)] = float(k)

        y_sv = np.asarray(self.support_y, dtype=float)
        alpha = np.asarray(self.alpha_counts, dtype=float)

        # Build the row K(x_i, support_x)
        row = np.asarray([self._train_kernel_cache[(i, j)] for j in support_train_idx], dtype=float)
        return float(scale * np.dot(row, alpha * y_sv))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.support_x:
            return np.zeros(X.shape[0], dtype=float)

        T = max(1, self._T)
        scale = 1.0 / (self.lambda_reg * T)

        K = self.kernel.evaluate(x_vec=X, y_vec=np.asarray(self.support_x))
        K = np.asarray(K, dtype=float)

        y_sv = np.asarray(self.support_y, dtype=float)
        alpha = np.asarray(self.alpha_counts, dtype=float)

        return scale * (K * (alpha * y_sv)[None, :]).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)
