from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def expand_polynomial_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    degree: int = 2,
    include_bias: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand features using a simple polynomial basis.

    For d input features, degree-2 expansion yields ~O(d^2) features (including
    squares and pairwise interactions), which is enough to test the hypothesis
    that richer classical structure changes the kernel-geometry story.

    Returns (X_train_expanded, X_test_expanded, feature_names).
    """

    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=int(degree), include_bias=bool(include_bias))
    X_train_e = poly.fit_transform(X_train)
    X_test_e = poly.transform(X_test)
    names = poly.get_feature_names_out()
    return X_train_e, X_test_e, names


@dataclass(frozen=True)
class PCAResult:
    X_train: np.ndarray
    X_test: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int


def make_pca_pipeline(n_components: int, *, for_quantum: bool) -> Pipeline:
    if for_quantum:
        # Quantum feature maps typically assume angles in [0, pi] or [0, 2pi].
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    else:
        scaler = StandardScaler()

    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("scale", scaler),
        ]
    )


def fit_transform_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_components: int,
    for_quantum: bool,
) -> PCAResult:
    pipe = make_pca_pipeline(n_components=n_components, for_quantum=for_quantum)
    X_train_t = pipe.fit_transform(X_train)
    X_test_t = pipe.transform(X_test)

    pca = pipe.named_steps["pca"]
    return PCAResult(
        X_train=X_train_t,
        X_test=X_test_t,
        explained_variance_ratio=pca.explained_variance_ratio_.copy(),
        n_components=n_components,
    )


def standardize_full_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def pad_or_trim_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] > target_dim:
        return X[:, :target_dim]

    pad_width = target_dim - X.shape[1]
    return np.pad(X, pad_width=((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
