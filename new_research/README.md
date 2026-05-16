# QML for Fake News — Research Experiments

This folder contains a cleaned, reproducible experiment pipeline for:
- Quantum-kernel SVM (Pegasos-style training) with multiple feature maps.
- Kernel diagnostics (kernel alignment, spectrum/spectral gap, conditioning).
- Honest classical baselines (LogReg / RF / SVM-RBF) on the same features.
- PCA bottleneck study: explained variance vs accuracy vs qubits.
- Noise robustness (depolarizing noise) for NISQ relevance.

## Quickstart
1) Put the dataset CSV at:
- `new_research/data/facebook-fact-check.csv`

2) Install dependencies (from repo root):
- `c:/Users/kriti/Downloads/QML-paper/.venv/Scripts/python.exe -m pip install -r new_research/requirements.txt`

3) Run a quick end-to-end pass:
- `c:/Users/kriti/Downloads/QML-paper/.venv/Scripts/python.exe new_research/run_all.py --quick`

Outputs are written to `new_research/outputs/` including an auto-updated draft paper:
- `new_research/outputs/paper_draft.docx`
