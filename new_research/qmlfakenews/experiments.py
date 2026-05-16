from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_breast_cancer

from .baselines import baseline_results_to_dataframe, run_classical_baselines, run_classical_baselines_cv
from .data import engineer_facebook_factcheck_features, load_facebook_factcheck_csv, make_train_test_split, preprocess_facebook_factcheck
from .diagnostics import kernel_target_alignment, spectrum_stats
from .pegasos import KernelPegasosSVM
from .preprocess import expand_polynomial_features, fit_transform_pca, pad_or_trim_features, standardize_full_features
from .quantum_kernel import QuantumKernelFactory


def run_all(*, data_csv: Path, outputs_dir: Path, seed: int, quick: bool, preset: str = "standard", verbose: bool = True) -> dict:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"Loading dataset from: {data_csv}")
    df_raw = load_facebook_factcheck_csv(data_csv)
    X, y, feature_names = preprocess_facebook_factcheck(df_raw)

    # Harder engineered representation (no text available in this dataset): one-hot + date + log/ratio features.
    X_eng, y_eng, eng_names = engineer_facebook_factcheck_features(df_raw, top_k_categories=20)

    log(f"Preprocessed: n={X.shape[0]}, d={X.shape[1]}")

    split = make_train_test_split(X, y, feature_names, test_size=0.25, random_seed=seed)
    split_eng = make_train_test_split(X_eng, y_eng, eng_names, test_size=0.25, random_seed=seed)

    # Stronger feature setting (Option B): polynomial expansion to 20+ dims.
    # This does not change the dataset; it increases feature complexity deterministically.
    X_train_poly, X_test_poly, poly_names = expand_polynomial_features(split.X_train, split.X_test, degree=2)

    # Classical baselines: full features
    log("Running classical baselines (full features)...")
    X_train_full, X_test_full = standardize_full_features(split.X_train, split.X_test)
    baselines_full = baseline_results_to_dataframe(run_classical_baselines(X_train_full, split.y_train, X_test_full, split.y_test))
    baselines_full.to_csv(outputs_dir / "tables" / "baselines_full.csv", index=False)

    baselines_full_cv = run_classical_baselines_cv(split.X_train, split.y_train, seed=seed, n_splits=5)
    baselines_full_cv.to_csv(outputs_dir / "tables" / "baselines_full_cv.csv", index=False)

    # Classical baselines: PCA 2D (same as quantum entrypoint)
    log("Running classical baselines (PCA=2)...")
    pca2 = fit_transform_pca(split.X_train, split.X_test, n_components=2, for_quantum=False)
    baselines_2d = baseline_results_to_dataframe(run_classical_baselines(pca2.X_train, split.y_train, pca2.X_test, split.y_test))
    baselines_2d.to_csv(outputs_dir / "tables" / "baselines_pca2.csv", index=False)

    # Classical baselines: polynomial-expanded full features (20+ dims)
    log("Running classical baselines (poly2 expanded features)...")
    X_train_poly_s, X_test_poly_s = standardize_full_features(X_train_poly, X_test_poly)
    baselines_poly = baseline_results_to_dataframe(run_classical_baselines(X_train_poly_s, split.y_train, X_test_poly_s, split.y_test))
    baselines_poly.to_csv(outputs_dir / "tables" / "baselines_poly2_full.csv", index=False)

    # Classical baselines: engineered features (harder, higher-dimensional)
    log("Running classical baselines (engineered features)...")
    X_train_eng_s, X_test_eng_s = standardize_full_features(split_eng.X_train, split_eng.X_test)
    baselines_eng = baseline_results_to_dataframe(run_classical_baselines(X_train_eng_s, split_eng.y_train, X_test_eng_s, split_eng.y_test))
    baselines_eng.to_csv(outputs_dir / "tables" / "baselines_engineered_full.csv", index=False)

    # PCA(2) variance on engineered features (helps address "PCA(2) keeps everything" critique)
    pca2_eng = fit_transform_pca(split_eng.X_train, split_eng.X_test, n_components=2, for_quantum=False)

    # CV on PCA(2) features (fit PCA inside CV would be more correct, but this still
    # provides a useful stability check on the reduced representation).
    baselines_pca2_cv = run_classical_baselines_cv(pca2.X_train, split.y_train, seed=seed, n_splits=5)
    baselines_pca2_cv.to_csv(outputs_dir / "tables" / "baselines_pca2_cv.csv", index=False)

    # Quantum experiments (feature maps + diagnostics + robustness)
    kernel_diag_rows: list[dict] = []
    qmap_rows: list[dict] = []
    tradeoff_rows: list[dict] = []
    noise_rows: list[dict] = []
    convergence_rows: list[dict] = []
    geometry_rows: list[dict] = []

    feature_maps = ["z", "zz", "pauli_xyz", "ry_custom"]
    if preset not in {"standard", "full"}:
        raise ValueError("preset must be one of: 'standard', 'full'")

    if quick:
        pca_grid = (2,)
        qubits_grid = (2,)
        noise_grid = (0.0,)
        iters_main = 60
        iters_noise = 40
        diag_n = 200
        reps_grid = (2,)
        viz_n = 50
        eval_subset_n = 200
        shots_noise = 512
    elif preset == "standard":
        # Paper-safe defaults for modest machines (avoid large sweeps).
        pca_grid = (2, 3, 4)
        qubits_grid = (2, 3)
        noise_grid = (0.0, 0.01, 0.03, 0.05, 0.1)
        iters_main = 90
        iters_noise = 40
        diag_n = 250
        reps_grid = (1, 2, 3)
        viz_n = 50
        eval_subset_n = 250
        shots_noise = 512
        seeds_grid = tuple(range(10))
    else:
        # Deeper sweeps for final paper runs
        pca_grid = (2, 3, 4)
        qubits_grid = (2, 3, 4)
        noise_grid = (0.0, 0.001, 0.005, 0.01, 0.02)
        iters_main = 200
        iters_noise = 120
        diag_n = 400
        reps_grid = (1, 2, 4, 6)
        viz_n = 70
        eval_subset_n = 300
        shots_noise = 2048
        seeds_grid = tuple(range(10))

    diag_n = min(diag_n, split.X_train.shape[0])
    viz_n = min(viz_n, split.X_train.shape[0])

    sss = StratifiedShuffleSplit(n_splits=1, train_size=diag_n, random_state=seed)
    diag_idx, _ = next(sss.split(split.X_train, split.y_train))

    sss_viz = StratifiedShuffleSplit(n_splits=1, train_size=viz_n, random_state=seed + 1)
    viz_idx, _ = next(sss_viz.split(split.X_train, split.y_train))

    # Feature-map comparison (2D entrypoint): accuracy + sparsity + kernel diagnostics
    log("Running quantum feature-map comparison (2 qubits, PCA=2)...")
    for fmap in feature_maps:
        try:
            pca_q = fit_transform_pca(split.X_train, split.X_test, n_components=2, for_quantum=True)
            Xq_train = pad_or_trim_features(pca_q.X_train, 2)
            Xq_test = pad_or_trim_features(pca_q.X_test, 2)

            # Train Pegasos on the full train split
            kernel = QuantumKernelFactory(feature_map_name=fmap, num_qubits=2, reps=2, seed=seed, noise_prob=0.0).build()
            pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=iters_main, seed=seed)
            hist = pegasos.fit(
                Xq_train,
                split.y_train,
                X_test=Xq_test,
                y_test=split.y_test,
                eval_every=20,
                eval_subset_n=eval_subset_n,
            )
            test_acc = float(np.mean(pegasos.predict(Xq_test) == split.y_test))

            qmap_rows.append(
                {
                    "feature_map": fmap,
                    "test_accuracy": test_acc,
                    "n_support": pegasos.n_support_vectors,
                }
            )

            # Diagnostics on a stratified subset to avoid O(n^2) blow-ups
            X_diag = Xq_train[diag_idx]
            y_diag = split.y_train[diag_idx]
            K = np.asarray(kernel.evaluate(x_vec=X_diag, y_vec=X_diag), dtype=float)

            align = kernel_target_alignment(K, y_diag, centered=True)
            spec = spectrum_stats(K, centered=True)

            # Mandatory visuals: kernel heatmaps + eigen spectra on a smaller subset
            try:
                from .plotting import save_eigenspectrum, save_kernel_heatmap

                X_viz = Xq_train[viz_idx]
                y_viz = split.y_train[viz_idx]
                K_viz = np.asarray(kernel.evaluate(x_vec=X_viz, y_vec=X_viz), dtype=float)
                spec_viz = spectrum_stats(K_viz, centered=True)

                save_kernel_heatmap(
                    K_viz,
                    outputs_dir / "figures" / f"kernel_heatmap_{fmap}.png",
                    title=f"{fmap} (PCA=2, qubits=2, reps=2)",
                )
                save_eigenspectrum(
                    spec_viz.eigenvalues,
                    outputs_dir / "figures" / f"eigenspectrum_{fmap}.png",
                    title=f"{fmap} centered Gram spectrum (n={len(y_viz)})",
                )
            except Exception:
                # Visualization should not fail the experiment run
                pass

            kernel_diag_rows.append(
                {
                    "feature_map": fmap,
                    "kernel_alignment": align,
                    "spectral_gap": spec.spectral_gap,
                    "condition_number": spec.condition_number,
                    "effective_rank": spec.effective_rank,
                    "leading_eigen_ratio": spec.leading_eigen_ratio,
                    "diag_subset_n": int(diag_n),
                }
            )

            geometry_rows.append(
                {
                    "dataset": "facebook_factcheck",
                    "source": "feature_map",
                    "feature_map": fmap,
                    "reps": 2,
                    "pca_components": 2,
                    "qubits": 2,
                    "accuracy": test_acc,
                    "n_support": float(pegasos.n_support_vectors),
                    "kernel_alignment": align,
                    "spectral_gap": spec.spectral_gap,
                    "condition_number": spec.condition_number,
                    "effective_rank": spec.effective_rank,
                    "leading_eigen_ratio": spec.leading_eigen_ratio,
                    "diag_subset_n": int(diag_n),
                }
            )

            # Capture convergence curve for the best candidate map (Pauli)
            if fmap == "pauli_xyz":
                for it, ns, tr, te in zip(hist.iterations, hist.n_support, hist.train_accuracy, hist.test_accuracy):
                    convergence_rows.append(
                        {
                            "iteration": it,
                            "n_support": ns,
                            "train_accuracy": tr,
                            "test_accuracy": te,
                        }
                    )
        except Exception as exc:
            kernel_diag_rows.append(
                {
                    "feature_map": fmap,
                    "kernel_alignment": np.nan,
                    "spectral_gap": np.nan,
                    "condition_number": np.nan,
                    "diag_subset_n": int(diag_n),
                    "error": str(exc),
                }
            )
            qmap_rows.append(
                {
                    "feature_map": fmap,
                    "test_accuracy": np.nan,
                    "n_support": np.nan,
                    "error": str(exc),
                }
            )

    kernel_diagnostics_df = pd.DataFrame(kernel_diag_rows)
    kernel_diagnostics_df.to_csv(outputs_dir / "tables" / "kernel_diagnostics.csv", index=False)

    qmap_df = pd.DataFrame(qmap_rows)
    qmap_df.to_csv(outputs_dir / "tables" / "quantum_featuremap_comparison.csv", index=False)

    # Feature-map depth sweep (reps) @ fixed PCA=2, qubits=2
    log(f"Running feature-map depth sweep (reps={reps_grid})...")
    for fmap in ["z", "zz", "pauli_xyz"]:
        for reps in reps_grid:
            try:
                pca_q = fit_transform_pca(split.X_train, split.X_test, n_components=2, for_quantum=True)
                Xq_train = pad_or_trim_features(pca_q.X_train, 2)
                Xq_test = pad_or_trim_features(pca_q.X_test, 2)

                kernel = QuantumKernelFactory(feature_map_name=fmap, num_qubits=2, reps=int(reps), seed=seed, noise_prob=0.0).build()
                pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=max(60, iters_main // 2), seed=seed)
                pegasos.fit(Xq_train, split.y_train, X_test=Xq_test, y_test=split.y_test, eval_every=0)
                test_acc = float(np.mean(pegasos.predict(Xq_test) == split.y_test))

                X_diag = Xq_train[diag_idx]
                y_diag = split.y_train[diag_idx]
                K = np.asarray(kernel.evaluate(x_vec=X_diag, y_vec=X_diag), dtype=float)
                align = kernel_target_alignment(K, y_diag, centered=True)
                spec = spectrum_stats(K, centered=True)

                geometry_rows.append(
                    {
                        "dataset": "facebook_factcheck",
                        "source": "depth_sweep",
                        "feature_map": fmap,
                        "reps": int(reps),
                        "pca_components": 2,
                        "qubits": 2,
                        "accuracy": test_acc,
                        "n_support": float(pegasos.n_support_vectors),
                        "kernel_alignment": align,
                        "spectral_gap": spec.spectral_gap,
                        "condition_number": spec.condition_number,
                        "effective_rank": spec.effective_rank,
                        "leading_eigen_ratio": spec.leading_eigen_ratio,
                        "diag_subset_n": int(diag_n),
                    }
                )

            except Exception as exc:
                geometry_rows.append(
                    {
                        "dataset": "facebook_factcheck",
                        "source": "depth_sweep",
                        "feature_map": fmap,
                        "reps": int(reps),
                        "pca_components": 2,
                        "qubits": 2,
                        "accuracy": np.nan,
                        "n_support": np.nan,
                        "kernel_alignment": np.nan,
                        "spectral_gap": np.nan,
                        "condition_number": np.nan,
                        "effective_rank": np.nan,
                        "leading_eigen_ratio": np.nan,
                        "diag_subset_n": int(diag_n),
                        "error": str(exc),
                    }
                )

    # PCA-vs-qubits tradeoff (quantum + classical reference)
    log(f"Running PCA-vs-qubits tradeoff sweep (PCA={pca_grid}, qubits={qubits_grid})...")
    for pca_components in pca_grid:
        pca_q = fit_transform_pca(split.X_train, split.X_test, n_components=pca_components, for_quantum=True)
        evr_sum = float(np.sum(pca_q.explained_variance_ratio))

        # Classical reference on same PCA dims (standardized)
        pca_c = fit_transform_pca(split.X_train, split.X_test, n_components=pca_components, for_quantum=False)
        base = run_classical_baselines(pca_c.X_train, split.y_train, pca_c.X_test, split.y_test)
        # pick SVM_RBF as headline classical kernel baseline
        svm_rbf = next((r for r in base if r.model_name == "SVM_RBF"), None)
        if svm_rbf is not None:
            tradeoff_rows.append(
                {
                    "model": "SVM_RBF",
                    "pca_components": pca_components,
                    "qubits": 0,
                    "explained_variance": evr_sum,
                    "accuracy": svm_rbf.accuracy,
                }
            )

        # Quantum (Pauli): vary qubits independently of PCA dimension (pad/trim)
        for qubits in qubits_grid:
            try:
                Xq_train = pad_or_trim_features(pca_q.X_train, qubits)
                Xq_test = pad_or_trim_features(pca_q.X_test, qubits)

                kernel = QuantumKernelFactory(feature_map_name="pauli_xyz", num_qubits=qubits, reps=2, seed=seed, noise_prob=0.0).build()
                pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=iters_main, seed=seed)
                pegasos.fit(Xq_train, split.y_train, X_test=Xq_test, y_test=split.y_test, eval_every=0)

                acc = float(np.mean(pegasos.predict(Xq_test) == split.y_test))
                # Diagnostics for correlation analysis (subset)
                X_diag = Xq_train[diag_idx]
                y_diag = split.y_train[diag_idx]
                K = np.asarray(kernel.evaluate(x_vec=X_diag, y_vec=X_diag), dtype=float)
                align = kernel_target_alignment(K, y_diag, centered=True)
                spec = spectrum_stats(K, centered=True)

                tradeoff_rows.append(
                    {
                        "model": "QKernel_Pauli_Pegasos",
                        "pca_components": pca_components,
                        "qubits": qubits,
                        "explained_variance": evr_sum,
                        "accuracy": acc,
                        "n_support": pegasos.n_support_vectors,
                    }
                )

                geometry_rows.append(
                    {
                        "dataset": "facebook_factcheck",
                        "source": "tradeoff",
                        "feature_map": "pauli_xyz",
                        "reps": 2,
                        "pca_components": pca_components,
                        "qubits": int(qubits),
                        "accuracy": acc,
                        "n_support": float(pegasos.n_support_vectors),
                        "kernel_alignment": align,
                        "spectral_gap": spec.spectral_gap,
                        "condition_number": spec.condition_number,
                        "effective_rank": spec.effective_rank,
                        "leading_eigen_ratio": spec.leading_eigen_ratio,
                        "diag_subset_n": int(diag_n),
                    }
                )

            except Exception as exc:
                tradeoff_rows.append(
                    {
                        "model": "QKernel_Pauli_Pegasos",
                        "pca_components": pca_components,
                        "qubits": qubits,
                        "explained_variance": evr_sum,
                        "accuracy": np.nan,
                        "n_support": np.nan,
                        "error": str(exc),
                    }
                )

    pca_tradeoff_df = pd.DataFrame(tradeoff_rows)
    pca_tradeoff_df.to_csv(outputs_dir / "tables" / "pca_qubits_tradeoff.csv", index=False)

    # Geometry ↔ performance dataset + simple correlation summary
    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(outputs_dir / "tables" / "kernel_geometry_performance.csv", index=False)

    corr_rows: list[dict] = []
    g_use = geometry_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "accuracy",
            "kernel_alignment",
            "condition_number",
            "spectral_gap",
            "effective_rank",
            "leading_eigen_ratio",
        ]
    )
    if not g_use.empty:
        for metric in ["kernel_alignment", "condition_number", "spectral_gap", "effective_rank", "leading_eigen_ratio"]:
            pearson = float(g_use[[metric, "accuracy"]].corr(method="pearson").iloc[0, 1])
            spearman = float(g_use[[metric, "accuracy"]].corr(method="spearman").iloc[0, 1])
            corr_rows.append({"metric": metric, "target": "accuracy", "pearson_r": pearson, "spearman_r": spearman, "n": int(g_use.shape[0])})
    pd.DataFrame(corr_rows).to_csv(outputs_dir / "tables" / "kernel_geometry_correlations.csv", index=False)

    # Statistical significance / stability (10 seeds): focus on one fair headline comparison.
    if not quick:
        log(f"Running 10-seed stability sweep (seeds={list(seeds_grid)})...")
        seed_rows: list[dict] = []
        for s in seeds_grid:
            try:
                split_s = make_train_test_split(X, y, feature_names, test_size=0.25, random_seed=int(s))
                # PCA(2) for fair comparison
                pca_c = fit_transform_pca(split_s.X_train, split_s.X_test, n_components=2, for_quantum=False)
                base = run_classical_baselines(pca_c.X_train, split_s.y_train, pca_c.X_test, split_s.y_test)
                svm_rbf = next((r for r in base if r.model_name == "SVM_RBF"), None)
                svm_acc = float(svm_rbf.accuracy) if svm_rbf is not None else float("nan")
                svm_ns = float(svm_rbf.extra.get("n_support")) if svm_rbf is not None else float("nan")

                pca_q = fit_transform_pca(split_s.X_train, split_s.X_test, n_components=2, for_quantum=True)
                Xq_train = pad_or_trim_features(pca_q.X_train, 2)
                Xq_test = pad_or_trim_features(pca_q.X_test, 2)
                kernel = QuantumKernelFactory(feature_map_name="pauli_xyz", num_qubits=2, reps=2, seed=int(s), noise_prob=0.0).build()
                pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=max(60, iters_main // 2), seed=int(s))
                pegasos.fit(Xq_train, split_s.y_train, X_test=Xq_test, y_test=split_s.y_test, eval_every=0)
                q_acc = float(np.mean(pegasos.predict(Xq_test) == split_s.y_test))
                q_ns = float(pegasos.n_support_vectors)

                seed_rows.append(
                    {
                        "seed": int(s),
                        "svm_rbf_accuracy": svm_acc,
                        "svm_rbf_n_support": svm_ns,
                        "q_pauli_accuracy": q_acc,
                        "q_pauli_n_support": q_ns,
                        "compression_ratio_svm_over_q": (svm_ns / max(1.0, q_ns)) if np.isfinite(svm_ns) and np.isfinite(q_ns) else float("nan"),
                    }
                )
            except Exception as exc:
                seed_rows.append({"seed": int(s), "error": str(exc)})

        seed_df = pd.DataFrame(seed_rows)
        seed_df.to_csv(outputs_dir / "tables" / "seed_sweep_pca2.csv", index=False)

        # Summary stats + simple bootstrap CI for mean differences/ratios (no SciPy dependency)
        def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, rng_seed: int = 123) -> tuple[float, float]:
            rng = np.random.default_rng(rng_seed)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return (float("nan"), float("nan"))
            boots = []
            for _ in range(int(n_boot)):
                samp = rng.choice(values, size=values.size, replace=True)
                boots.append(float(np.mean(samp)))
            lo = float(np.quantile(boots, alpha / 2))
            hi = float(np.quantile(boots, 1 - alpha / 2))
            return (lo, hi)

        use = seed_df.replace([np.inf, -np.inf], np.nan)
        use = use.dropna(subset=["svm_rbf_accuracy", "q_pauli_accuracy", "svm_rbf_n_support", "q_pauli_n_support", "compression_ratio_svm_over_q"])
        if not use.empty:
            diff_acc = (use["q_pauli_accuracy"].to_numpy(dtype=float) - use["svm_rbf_accuracy"].to_numpy(dtype=float))
            ratio = use["compression_ratio_svm_over_q"].to_numpy(dtype=float)
            lo_d, hi_d = _bootstrap_ci(diff_acc)
            lo_r, hi_r = _bootstrap_ci(ratio)
            summary_rows = [
                {
                    "metric": "accuracy_diff_q_minus_svm",
                    "mean": float(np.mean(diff_acc)),
                    "std": float(np.std(diff_acc, ddof=1)) if diff_acc.size > 1 else 0.0,
                    "ci95_low": lo_d,
                    "ci95_high": hi_d,
                    "n": int(diff_acc.size),
                },
                {
                    "metric": "compression_ratio_svm_over_q",
                    "mean": float(np.mean(ratio)),
                    "std": float(np.std(ratio, ddof=1)) if ratio.size > 1 else 0.0,
                    "ci95_low": lo_r,
                    "ci95_high": hi_r,
                    "n": int(ratio.size),
                },
            ]
            pd.DataFrame(summary_rows).to_csv(outputs_dir / "tables" / "seed_sweep_summary.csv", index=False)

    # Auxiliary dataset (Option A): small, built-in tabular benchmark to test generality
    # without requiring downloads. Keep this lightweight.
    if not quick:
        log("Running auxiliary dataset (breast cancer) sanity check...")
        try:
            bc = load_breast_cancer()
            X_bc = bc.data.astype(float)
            y_bc = bc.target.astype(int)
            y_bc = np.where(y_bc == 1, 1, -1)

            split_bc = make_train_test_split(X_bc, y_bc, list(map(str, bc.feature_names)), test_size=0.25, random_seed=seed)

            # Classical baselines on full standardized features
            X_bc_tr, X_bc_te = standardize_full_features(split_bc.X_train, split_bc.X_test)
            bc_full = baseline_results_to_dataframe(run_classical_baselines(X_bc_tr, split_bc.y_train, X_bc_te, split_bc.y_test))
            bc_full.to_csv(outputs_dir / "tables" / "aux_breastcancer_baselines_full.csv", index=False)

            # PCA(2) baselines
            bc_pca2 = fit_transform_pca(split_bc.X_train, split_bc.X_test, n_components=2, for_quantum=False)
            bc_pca2_base = baseline_results_to_dataframe(run_classical_baselines(bc_pca2.X_train, split_bc.y_train, bc_pca2.X_test, split_bc.y_test))
            bc_pca2_base.to_csv(outputs_dir / "tables" / "aux_breastcancer_baselines_pca2.csv", index=False)

            # Quantum kernel (very small sweep)
            bc_pca2_q = fit_transform_pca(split_bc.X_train, split_bc.X_test, n_components=2, for_quantum=True)
            Xq_tr = pad_or_trim_features(bc_pca2_q.X_train, 2)
            Xq_te = pad_or_trim_features(bc_pca2_q.X_test, 2)

            aux_rows: list[dict] = []
            for fmap in ["z", "zz", "pauli_xyz"]:
                kernel = QuantumKernelFactory(feature_map_name=fmap, num_qubits=2, reps=2, seed=seed, noise_prob=0.0).build()
                pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=max(60, iters_main // 2), seed=seed)
                pegasos.fit(Xq_tr, split_bc.y_train, X_test=Xq_te, y_test=split_bc.y_test, eval_every=0)
                acc = float(np.mean(pegasos.predict(Xq_te) == split_bc.y_test))

                # Diagnostics on a small subset
                diag_n_bc = min(diag_n, Xq_tr.shape[0])
                sss_bc = StratifiedShuffleSplit(n_splits=1, train_size=diag_n_bc, random_state=seed)
                diag_idx_bc, _ = next(sss_bc.split(Xq_tr, split_bc.y_train))
                X_diag_bc = Xq_tr[diag_idx_bc]
                y_diag_bc = split_bc.y_train[diag_idx_bc]
                K_bc = np.asarray(kernel.evaluate(x_vec=X_diag_bc, y_vec=X_diag_bc), dtype=float)
                align_bc = kernel_target_alignment(K_bc, y_diag_bc, centered=True)
                spec_bc = spectrum_stats(K_bc, centered=True)

                aux_rows.append({"feature_map": fmap, "test_accuracy": acc, "n_support": pegasos.n_support_vectors})
                geometry_rows.append(
                    {
                        "dataset": "breast_cancer",
                        "source": "aux_feature_map",
                        "feature_map": fmap,
                        "reps": 2,
                        "pca_components": 2,
                        "qubits": 2,
                        "accuracy": acc,
                        "n_support": float(pegasos.n_support_vectors),
                        "kernel_alignment": align_bc,
                        "spectral_gap": spec_bc.spectral_gap,
                        "condition_number": spec_bc.condition_number,
                        "effective_rank": spec_bc.effective_rank,
                        "leading_eigen_ratio": spec_bc.leading_eigen_ratio,
                        "diag_subset_n": int(diag_n_bc),
                    }
                )

            pd.DataFrame(aux_rows).to_csv(outputs_dir / "tables" / "aux_breastcancer_quantum_featuremaps.csv", index=False)
        except Exception as exc:
            pd.DataFrame([{"error": str(exc)}]).to_csv(outputs_dir / "tables" / "aux_breastcancer_error.csv", index=False)

    # Noise robustness (Pauli @ 2 qubits)
    log(f"Running noise robustness sweep (p in {noise_grid})...")
    try:
        pca_q = fit_transform_pca(split.X_train, split.X_test, n_components=2, for_quantum=True)
        Xq_train = pad_or_trim_features(pca_q.X_train, 2)
        Xq_test = pad_or_trim_features(pca_q.X_test, 2)

        # Small subset for kernel-level stats (kept tiny for runtime)
        noise_viz_n = min(30, Xq_train.shape[0])
        sss_noise = StratifiedShuffleSplit(n_splits=1, train_size=noise_viz_n, random_state=seed + 7)
        noise_idx, _ = next(sss_noise.split(Xq_train, split.y_train))
        X_noise = Xq_train[noise_idx]

        for p in noise_grid:
            kernel = QuantumKernelFactory(
                feature_map_name="pauli_xyz",
                num_qubits=2,
                reps=2,
                seed=seed,
                noise_prob=float(p),
                shots=int(shots_noise),
            ).build()
            pegasos = KernelPegasosSVM(kernel, lambda_reg=0.02, iterations=iters_noise, seed=seed)
            pegasos.fit(Xq_train, split.y_train, X_test=Xq_test, y_test=split.y_test, eval_every=0)
            acc = float(np.mean(pegasos.predict(Xq_test) == split.y_test))

            # Kernel-level evidence: gram statistics should degrade as p increases
            try:
                K_noise = np.asarray(kernel.evaluate(x_vec=X_noise, y_vec=X_noise), dtype=float)
                k_min = float(np.min(K_noise))
                k_max = float(np.max(K_noise))
                k_mean = float(np.mean(K_noise))
            except Exception:
                k_min, k_max, k_mean = float("nan"), float("nan"), float("nan")

            noise_rows.append(
                {
                    "noise_prob": float(p),
                    "accuracy": acc,
                    "k_min": k_min,
                    "k_mean": k_mean,
                    "k_max": k_max,
                    "shots": int(shots_noise),
                    "subset_n": int(noise_viz_n),
                }
            )
    except Exception as exc:
        noise_rows.append({"noise_prob": np.nan, "accuracy": np.nan, "error": str(exc)})

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(outputs_dir / "tables" / "noise_robustness.csv", index=False)

    convergence_df = pd.DataFrame(convergence_rows)
    convergence_df.to_csv(outputs_dir / "tables" / "pegasos_convergence.csv", index=False)

    log("Done.")

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_poly2": int(X_train_poly.shape[1]),
        "n_features_engineered": int(X_eng.shape[1]),
        "pca2_explained_variance": float(np.sum(pca2.explained_variance_ratio)),
        "pca2_engineered_explained_variance": float(np.sum(pca2_eng.explained_variance_ratio)),
        "baselines_full_path": str(outputs_dir / "tables" / "baselines_full.csv"),
        "baselines_full_cv_path": str(outputs_dir / "tables" / "baselines_full_cv.csv"),
        "baselines_pca2_path": str(outputs_dir / "tables" / "baselines_pca2.csv"),
        "baselines_pca2_cv_path": str(outputs_dir / "tables" / "baselines_pca2_cv.csv"),
        "quantum_featuremap_path": str(outputs_dir / "tables" / "quantum_featuremap_comparison.csv"),
        "kernel_diagnostics_path": str(outputs_dir / "tables" / "kernel_diagnostics.csv"),
        "geometry_performance_path": str(outputs_dir / "tables" / "kernel_geometry_performance.csv"),
        "geometry_correlations_path": str(outputs_dir / "tables" / "kernel_geometry_correlations.csv"),
        "pca_tradeoff_path": str(outputs_dir / "tables" / "pca_qubits_tradeoff.csv"),
        "noise_path": str(outputs_dir / "tables" / "noise_robustness.csv"),
        "pegasos_convergence_path": str(outputs_dir / "tables" / "pegasos_convergence.csv"),
    }
