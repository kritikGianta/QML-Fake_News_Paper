from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    extra: dict


def baseline_results_to_dataframe(results: list[BaselineResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        rows.append(
            {
                "model": r.model_name,
                "accuracy": float(r.accuracy),
                "precision": float(r.precision),
                "recall": float(r.recall),
                "f1": float(r.f1),
                "n_support": float(r.extra.get("n_support")) if isinstance(r.extra, dict) and "n_support" in r.extra else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(precision_score(y_true, y_pred, zero_division=0, pos_label=1)),
        float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
    )


def run_classical_baselines(X_train, y_train, X_test, y_test) -> list[BaselineResult]:
    results: list[BaselineResult] = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc, prec, rec, f1 = _binary_metrics(y_test, y_pred)
    results.append(BaselineResult("LogReg", acc, prec, rec, f1, extra={}))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc, prec, rec, f1 = _binary_metrics(y_test, y_pred)
    results.append(BaselineResult("RandomForest", acc, prec, rec, f1, extra={}))

    # SVM RBF
    svm = SVC(kernel="rbf", gamma="scale", C=1.0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc, prec, rec, f1 = _binary_metrics(y_test, y_pred)
    results.append(
        BaselineResult(
            "SVM_RBF",
            acc,
            prec,
            rec,
            f1,
            extra={"n_support": int(np.sum(svm.n_support_))},
        )
    )

    return results


def run_classical_baselines_cv(X: np.ndarray, y: np.ndarray, *, seed: int = 42, n_splits: int = 5) -> pd.DataFrame:
    """Fold-safe baseline evaluation.

    Returns a dataframe with mean/std for each metric per model.
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models: dict[str, object] = {
        "LogReg": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        ),
        "SVM_RBF": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", gamma="scale", C=1.0)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    rows: list[dict] = []
    for name, model in models.items():
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc, prec, rec, f1 = _binary_metrics(y_te, y_pred)
            metrics["accuracy"].append(acc)
            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            metrics["f1"].append(f1)

        rows.append(
            {
                "model": name,
                "accuracy_mean": float(np.mean(metrics["accuracy"])),
                "accuracy_std": float(np.std(metrics["accuracy"], ddof=1)),
                "precision_mean": float(np.mean(metrics["precision"])),
                "precision_std": float(np.std(metrics["precision"], ddof=1)),
                "recall_mean": float(np.mean(metrics["recall"])),
                "recall_std": float(np.std(metrics["recall"], ddof=1)),
                "f1_mean": float(np.mean(metrics["f1"])),
                "f1_std": float(np.std(metrics["f1"], ddof=1)),
                "n_splits": int(n_splits),
            }
        )

    return pd.DataFrame(rows)
