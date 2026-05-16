from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_facebook_factcheck_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset CSV at {csv_path}. Place 'facebook-fact-check.csv' under new_research/data/."
        )
    return pd.read_csv(csv_path)


def preprocess_facebook_factcheck(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()

    cols_to_drop = ["Debate", "account_id", "post_id", "Post URL"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    if "share_count" in df.columns:
        df["share_count"] = df["share_count"].fillna(0)

    required_non_null = [c for c in ["reaction_count", "comment_count"] if c in df.columns]
    if required_non_null:
        df = df.dropna(subset=required_non_null)

    if "Rating" not in df.columns:
        raise ValueError("Expected a 'Rating' column in the dataset.")

    df = df[df["Rating"].astype(str).str.lower() != "no factual content"]

    label_map = {
        "mostly true": 1,
        "mostly false": -1,
        "mixture of true and false": -1,
    }
    df["Rating"] = df["Rating"].map(label_map)
    df = df.dropna(subset=["Rating"])

    categorical_cols = ["Category", "Page", "Date Published", "Post Type"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    y = df["Rating"].astype(int).to_numpy()
    feature_df = df.drop(columns=["Rating"])

    X = feature_df.to_numpy(dtype=float)
    feature_names = list(feature_df.columns)
    return X, y, feature_names


def engineer_facebook_factcheck_features(df: pd.DataFrame, *, top_k_categories: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature engineering to create a harder, higher-dimensional representation.

    Motivation: the raw tabular representation is low-dimensional and PCA(2) retains
    very high variance, which can make the task appear "too easy". This function
    builds a 20+ dimensional feature set using:
    - numeric counts + log1p transforms + simple ratios
    - calendar features from Date Published
    - one-hot encodings for Category and Post Type (with top-K grouping)

    Returns (X, y, feature_names) with y in {-1, +1}.
    """

    df = df.copy()

    # Keep the same label filtering as preprocess_facebook_factcheck
    if "Rating" not in df.columns:
        raise ValueError("Expected a 'Rating' column in the dataset.")

    df = df[df["Rating"].astype(str).str.lower() != "no factual content"]
    label_map = {
        "mostly true": 1,
        "mostly false": -1,
        "mixture of true and false": -1,
    }
    df["Rating"] = df["Rating"].map(label_map)
    df = df.dropna(subset=["Rating"])
    y = df["Rating"].astype(int).to_numpy()

    # Numeric primitives
    for c in ["share_count", "reaction_count", "comment_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Date features
    if "Date Published" in df.columns:
        dt = pd.to_datetime(df["Date Published"], errors="coerce")
        df["pub_year"] = dt.dt.year.fillna(0).astype(int)
        df["pub_month"] = dt.dt.month.fillna(0).astype(int)
        df["pub_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
    else:
        df["pub_year"] = 0
        df["pub_month"] = 0
        df["pub_dayofweek"] = 0

    # Log transforms and ratios (safe division)
    eps = 1e-9
    df["log_share"] = np.log1p(df["share_count"])
    df["log_reaction"] = np.log1p(df["reaction_count"])
    df["log_comment"] = np.log1p(df["comment_count"])
    df["reaction_per_share"] = df["reaction_count"] / (df["share_count"] + 1.0)
    df["comment_per_share"] = df["comment_count"] / (df["share_count"] + 1.0)
    df["comment_per_reaction"] = df["comment_count"] / (df["reaction_count"] + 1.0)
    df["engagement_total"] = df["share_count"] + df["reaction_count"] + df["comment_count"]
    df["log_engagement_total"] = np.log1p(df["engagement_total"])
    df["share_frac"] = df["share_count"] / (df["engagement_total"] + eps)
    df["reaction_frac"] = df["reaction_count"] / (df["engagement_total"] + eps)
    df["comment_frac"] = df["comment_count"] / (df["engagement_total"] + eps)

    # One-hot categorical: Category and Post Type (top-K grouping)
    def _topk_onehot(col: str, prefix: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(index=df.index)
        s = df[col].astype(str).fillna("(missing)")
        top = set(s.value_counts().head(int(top_k_categories)).index.tolist())
        s = s.where(s.isin(top), other="(other)")
        return pd.get_dummies(s, prefix=prefix, dummy_na=False)

    cat = _topk_onehot("Category", "cat")
    ptype = _topk_onehot("Post Type", "ptype")

    # Assemble feature matrix
    numeric_cols = [
        "share_count",
        "reaction_count",
        "comment_count",
        "log_share",
        "log_reaction",
        "log_comment",
        "reaction_per_share",
        "comment_per_share",
        "comment_per_reaction",
        "engagement_total",
        "log_engagement_total",
        "share_frac",
        "reaction_frac",
        "comment_frac",
        "pub_year",
        "pub_month",
        "pub_dayofweek",
    ]
    num = df[numeric_cols].astype(float)
    feat = pd.concat([num, cat, ptype], axis=1)

    X = feat.to_numpy(dtype=float)
    return X, y, list(feat.columns)


def make_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    test_size: float,
    random_seed: int,
) -> DatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
