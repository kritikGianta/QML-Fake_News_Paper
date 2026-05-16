from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running as: python new_research/run_all.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from qmlfakenews.experiments import run_all
from qmlfakenews.plotting import (
    save_accuracy_vs_qubits_plot,
    save_convergence_plot,
    save_geometry_scatter_plots,
    save_noise_plot,
    save_tradeoff_plot,
)
from qmlfakenews.report_docx import write_paper_docx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a reduced/fast set of experiments")
    parser.add_argument("--full", action="store_true", help="Run deeper sweeps (slower)")
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="Do not rerun experiments; regenerate plots + paper_draft.docx from existing outputs/ tables",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Optional path to the dataset CSV (defaults to new_research/data/facebook-fact-check.csv)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.data_csv:
        data_csv = Path(args.data_csv).expanduser().resolve()
    else:
        # Default location (preferred)
        data_csv = repo_root / "new_research" / "data" / "facebook-fact-check.csv"
        # Backward/alternate location
        if not data_csv.exists():
            alt = repo_root / "new_research" / "facebook-fact-check.csv"
            if alt.exists():
                data_csv = alt
    outputs_dir = repo_root / "new_research" / "outputs"

    if args.doc_only:
        # Reuse existing summary if available; otherwise fall back to empty.
        summary_path = outputs_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                summary = {}
        else:
            summary = {}
    else:
        preset = "full" if args.full else "standard"
        summary = run_all(
            data_csv=data_csv,
            outputs_dir=outputs_dir,
            seed=42,
            quick=bool(args.quick),
            preset=preset,
            verbose=not bool(args.quiet),
        )

    # Load produced tables
    baselines_2d = _safe_read_csv(outputs_dir / "tables" / "baselines_pca2.csv")
    baselines_2d_cv = _safe_read_csv(outputs_dir / "tables" / "baselines_pca2_cv.csv")
    baselines_full = _safe_read_csv(outputs_dir / "tables" / "baselines_full.csv")
    baselines_full_cv = _safe_read_csv(outputs_dir / "tables" / "baselines_full_cv.csv")
    baselines_poly2 = _safe_read_csv(outputs_dir / "tables" / "baselines_poly2_full.csv")
    baselines_engineered = _safe_read_csv(outputs_dir / "tables" / "baselines_engineered_full.csv")
    aux_bc_full = _safe_read_csv(outputs_dir / "tables" / "aux_breastcancer_baselines_full.csv")
    aux_bc_pca2 = _safe_read_csv(outputs_dir / "tables" / "aux_breastcancer_baselines_pca2.csv")
    aux_bc_qmaps = _safe_read_csv(outputs_dir / "tables" / "aux_breastcancer_quantum_featuremaps.csv")
    seed_sweep = _safe_read_csv(outputs_dir / "tables" / "seed_sweep_pca2.csv")
    seed_sweep_summary = _safe_read_csv(outputs_dir / "tables" / "seed_sweep_summary.csv")
    kernel_diagnostics = _safe_read_csv(outputs_dir / "tables" / "kernel_diagnostics.csv")
    geometry_perf = _safe_read_csv(outputs_dir / "tables" / "kernel_geometry_performance.csv")
    geometry_corr = _safe_read_csv(outputs_dir / "tables" / "kernel_geometry_correlations.csv")
    q_featuremaps = _safe_read_csv(outputs_dir / "tables" / "quantum_featuremap_comparison.csv")
    pca_tradeoff = _safe_read_csv(outputs_dir / "tables" / "pca_qubits_tradeoff.csv")
    noise_robustness = _safe_read_csv(outputs_dir / "tables" / "noise_robustness.csv")
    pegasos_convergence = _safe_read_csv(outputs_dir / "tables" / "pegasos_convergence.csv")
    text_welfake_qubit_scaling = _safe_read_csv(repo_root / "new_research" / "new_results" / "welfake" / "tables" / "qubit_scaling.csv")
    text_welfake_noise = _safe_read_csv(repo_root / "new_research" / "new_results" / "welfake" / "tables" / "noise_sweep.csv")
    text_liar_qubit_scaling = _safe_read_csv(repo_root / "new_research" / "new_results" / "liar" / "tables" / "qubit_scaling.csv")
    text_liar_noise = _safe_read_csv(repo_root / "new_research" / "new_results" / "liar" / "tables" / "noise_sweep.csv")

    fig_dir = Path("C:/Users/kriti/Downloads/QML-paper/New folder")
    # Plots
    if not pegasos_convergence.empty:
        save_convergence_plot(pegasos_convergence, fig_dir / "pegasos_convergence.png")
    if not pca_tradeoff.empty:
        save_tradeoff_plot(pca_tradeoff, fig_dir / "pca_tradeoff.png")
        save_accuracy_vs_qubits_plot(pca_tradeoff, fig_dir / "accuracy_vs_qubits.png")
    if not noise_robustness.empty:
        save_noise_plot(noise_robustness, fig_dir / "noise_robustness.png")
    if not geometry_perf.empty:
        save_geometry_scatter_plots(geometry_perf, fig_dir)

    headline_results = {
        "Samples": summary.get("n_samples"),
        "Raw features": summary.get("n_features"),
        "Poly(2) features": summary.get("n_features_poly2"),
        "Engineered features": summary.get("n_features_engineered"),
        "PCA(2) explained variance": summary.get("pca2_explained_variance"),
        "Engineered PCA(2) explained variance": summary.get("pca2_engineered_explained_variance"),
    }

    # Paper draft
    write_paper_docx(
        out_path=outputs_dir / "paper_draft.docx",
        headline_results=headline_results,
        baselines_2d=baselines_2d,
        baselines_2d_cv=baselines_2d_cv,
        baselines_full=baselines_full,
        baselines_full_cv=baselines_full_cv,
        baselines_poly2=baselines_poly2,
        baselines_engineered=baselines_engineered,
        aux_breastcancer_baselines_full=aux_bc_full,
        aux_breastcancer_baselines_pca2=aux_bc_pca2,
        aux_breastcancer_quantum_featuremaps=aux_bc_qmaps,
        seed_sweep=seed_sweep,
        seed_sweep_summary=seed_sweep_summary,
        quantum_featuremaps=q_featuremaps,
        kernel_diagnostics=kernel_diagnostics,
        geometry_performance=geometry_perf,
        geometry_correlations=geometry_corr,
        pca_tradeoff=pca_tradeoff,
        noise_robustness=noise_robustness,
        pegasos_convergence=pegasos_convergence,
        text_welfake_qubit_scaling=text_welfake_qubit_scaling,
        text_welfake_noise=text_welfake_noise,
        text_liar_qubit_scaling=text_liar_qubit_scaling,
        text_liar_noise=text_liar_noise,
    )

    (outputs_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
