from __future__ import annotations

from pathlib import Path

import matplotlib

# Avoid Tkinter backend issues on Windows during non-interactive runs.
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3.0,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _apply_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5, palette="colorblind")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def _finalize_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_convergence_plot(df: pd.DataFrame, out_path: Path) -> None:
    # df columns: iteration, n_support, train_accuracy, test_accuracy
    _apply_publication_style()
    fig, ax1 = plt.subplots(figsize=(8.0, 5.5))

    ax1.plot(df["iteration"], df["train_accuracy"], label="Train accuracy", color="#1f77b4")
    ax1.plot(df["iteration"], df["test_accuracy"], label="Test accuracy", color="#ff7f0e")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(df["iteration"], df["n_support"], label="Support vectors", color="#2ca02c", linestyle="--")
    ax2.set_ylabel("Support Vectors")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")

    _finalize_figure(fig, out_path)


def save_tradeoff_plot(df: pd.DataFrame, out_path: Path) -> None:
    # df columns: pca_components, qubits, explained_variance, accuracy
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sns.lineplot(
        data=df,
        x="explained_variance",
        y="accuracy",
        hue="model",
        style="pca_components",
        markers=True,
        ax=ax,
    )
    ax.set_xlabel("Explained Variance (sum)")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model / PCA dims", frameon=True)
    _finalize_figure(fig, out_path)


def save_noise_plot(df: pd.DataFrame, out_path: Path) -> None:
    # df columns: noise_prob, accuracy
    _apply_publication_style()
    use = df.copy() if df is not None else None
    if use is None or use.empty:
        return
    use = use.replace([np.inf, -np.inf], np.nan)
    use = use.dropna(subset=["noise_prob", "accuracy"])
    if use.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sns.lineplot(data=use, x="noise_prob", y="accuracy", marker="o", ax=ax)
    ax.set_xlabel("Depolarizing probability")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)

    # Optional: if kernel mean is present, plot it on a secondary axis to show kernel degradation.
    if "k_mean" in use.columns and use["k_mean"].notna().any():
        ax2 = ax.twinx()
        sns.lineplot(data=use, x="noise_prob", y="k_mean", marker="s", ax=ax2, color="C2")
        ax2.set_ylabel("Mean kernel similarity (subset)")

    ax.set_title("Noise robustness under depolarizing perturbations")
    _finalize_figure(fig, out_path)


def save_accuracy_vs_qubits_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Plot quantum accuracy as a function of qubit count.

    Expects `pca_qubits_tradeoff.csv`-style rows with columns:
    model, pca_components, qubits, accuracy.
    """

    if df is None or df.empty:
        return

    use = df.copy()
    if "model" in use.columns:
        use = use[use["model"].astype(str).str.startswith("QKernel_")]

    required = {"pca_components", "qubits", "accuracy"}
    if not required.issubset(set(use.columns)):
        return

    use = use.replace([np.inf, -np.inf], np.nan).dropna(subset=["qubits", "accuracy", "pca_components"])
    if use.empty:
        return

    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sns.lineplot(
        data=use,
        x="qubits",
        y="accuracy",
        hue="pca_components",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Quantum accuracy vs qubits")
    ax.legend(title="PCA components", frameon=True)
    _finalize_figure(fig, out_path)


def save_geometry_scatter_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate the mandatory scatter plots linking diagnostics to accuracy.

    Expects columns: kernel_alignment, condition_number, spectral_gap, accuracy.
    Optional columns: feature_map, reps, qubits.
    """

    if df is None or df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_publication_style()
    use = df.copy()
    use = use.replace([np.inf, -np.inf], np.nan)
    use = use.dropna(subset=["accuracy"])

    plots = [
        ("kernel_alignment", "alignment_vs_accuracy.png", "Kernel-target alignment", False),
        ("condition_number", "condition_vs_accuracy.png", "Condition number", True),
        ("spectral_gap", "spectralgap_vs_accuracy.png", "Spectral gap (λ1−λ2)", False),
        ("effective_rank", "effective_rank_vs_accuracy.png", "Effective rank (participation ratio)", False),
        ("leading_eigen_ratio", "leading_eigen_ratio_vs_accuracy.png", "Leading eigenvalue ratio (λ1 / Σλ+)", False),
    ]

    for xcol, fname, xlabel, logx in plots:
        if xcol not in use.columns:
            continue

        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        
        # Add regression line with confidence band and larger markers
        sns.regplot(
            data=use, 
            x=xcol, 
            y="accuracy", 
            ax=ax, 
            scatter_kws={"s": 120, "alpha": 0.8, "edgecolor": "w"},
            line_kws={"color": "darkred", "linewidth": 3.0, "alpha": 0.8}
        )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.0)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"{xlabel} vs accuracy")
        
        _finalize_figure(fig, out_dir / fname)


def save_kernel_heatmap(K: np.ndarray, out_path: Path, *, title: str | None = None) -> None:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sns.heatmap(K, cmap="viridis", square=True, cbar=True, ax=ax, cbar_kws={"shrink": 0.8})
    if title:
        ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    _finalize_figure(fig, out_path)


def save_eigenspectrum(evals: np.ndarray, out_path: Path, *, title: str | None = None) -> None:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    x = np.arange(1, len(evals) + 1)
    ax.plot(x, evals, marker="o", linewidth=1.2)
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("Eigenvalue")
    if title:
        ax.set_title(title)
    _finalize_figure(fig, out_path)
